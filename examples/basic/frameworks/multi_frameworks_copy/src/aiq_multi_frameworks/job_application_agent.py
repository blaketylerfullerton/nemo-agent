# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from typing import List, Dict, Any

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
import json
import resend
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class JobApplicationConfig(FunctionBaseConfig, name="job_application_agent"):
    job_type: str
    job_website_url: str


@register_function(config_type=JobApplicationConfig)
async def job_application_agent_as_tool(tool_config: JobApplicationConfig, builder: Builder):
    
    async def scrape_job_listings(url: str) -> List[Dict[str, Any]]:
        """
        Scrape job listings from a website URL and extract relevant information.
        This is a generic scraper that attempts to find job-related information.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            jobs = []
            
            # Parse domain name for company info
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '')
            
            # Look for common job listing patterns
            job_containers = []
            
            # Try different common selectors for job listings
            selectors = [
                '.job', '.job-listing', '.position', '.role', '.career',
                '[class*="job"]', '[class*="position"]', '[class*="career"]',
                'article', '.card', '.listing'
            ]
            
            for selector in selectors:
                containers = soup.select(selector)
                if containers:
                    job_containers = containers
                    break
            
            # If no specific job containers found, try to extract from general content
            if not job_containers:
                job_containers = [soup]
            
            for container in job_containers[:10]:  # Limit to first 10 jobs
                job = extract_job_info(container, domain, url)
                if job and job.get('job_description'):
                    jobs.append(job)
            
            # If no jobs found, create a generic entry from page content
            if not jobs:
                page_title = soup.find('title')
                title_text = page_title.get_text().strip() if page_title else f"Jobs at {domain}"
                
                # Try to find contact email
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                page_text = soup.get_text()
                emails = re.findall(email_pattern, page_text)
                contact_email = emails[0] if emails else f"careers@{domain}"
                
                jobs.append({
                    'name': f"Hiring Manager",
                    'company': domain.split('.')[0].title(),
                    'email': contact_email,
                    'website': url,
                    'job_description': title_text
                })
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []
    
    def extract_job_info(container: BeautifulSoup, domain: str, base_url: str) -> Dict[str, Any]:
        """
        Extract job information from a container element.
        """
        job = {
            'name': 'Hiring Manager',
            'company': domain.split('.')[0].title(),
            'email': f"careers@{domain}",
            'website': base_url,
            'job_description': ''
        }
        
        # Try to find job title
        title_selectors = ['h1', 'h2', 'h3', '.title', '[class*="title"]', '.job-title', '[class*="job-title"]']
        for selector in title_selectors:
            title_elem = container.select_one(selector)
            if title_elem:
                job['job_description'] = title_elem.get_text().strip()
                break
        
        # If no title found, use any text content
        if not job['job_description']:
            text_content = container.get_text().strip()
            if text_content:
                # Take first meaningful line as job description
                lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                job['job_description'] = lines[0] if lines else text_content[:100]
        
        # Try to find contact email in this container
        container_text = container.get_text()
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, container_text)
        if emails:
            job['email'] = emails[0]
        
        # Try to find company name
        company_selectors = ['.company', '[class*="company"]', '.employer', '[class*="employer"]']
        for selector in company_selectors:
            company_elem = container.select_one(selector)
            if company_elem:
                company_text = company_elem.get_text().strip()
                if company_text:
                    job['company'] = company_text
                break
        
        return job

    async def _arun(inputs: str) -> str:
        logger.info(f"[job_application_agent] RECEIVED INPUT: {inputs}")
        
        # Scrape jobs from the configured website
        jobs = await scrape_job_listings(tool_config.job_website_url)
        
        if not jobs:
            return f"No jobs found on {tool_config.job_website_url}"
        
        # Look for jobs matching the input (company name or job type)
        for job in jobs:
            if (job['company'].lower() in inputs.lower() or 
                inputs.lower() in job['job_description'].lower() or
                tool_config.job_type.lower() in job['job_description'].lower()):
                
                await send_email(
                    job['email'], 
                    "Job Application", 
                    f"Applying for position: {job['job_description']} at {job['company']} ({job['website']})"
                )
                output = f"Applied for position: {job['job_description']} at {job['company']} ({job['website']})"
                return output
        
        # If no specific match found, apply to the first job
        if jobs:
            job = jobs[0]
            await send_email(
                job['email'], 
                "Job Application", 
                f"Applying for position: {job['job_description']} at {job['company']} ({job['website']})"
            )
            output = f"Applied for position: {job['job_description']} at {job['company']} ({job['website']})"
            return output
        
        return f"No matching jobs found for '{inputs}' on {tool_config.job_website_url}"

    yield FunctionInfo.from_fn(_arun, description="scrape and apply for jobs from a website")


async def send_email(email: str, subject: str, body: str):
    resend.api_key = os.getenv("RESEND_API_KEY")
    params: resend.Emails.SendParams = {
        "from": "Soham <jobs@fluxolabs.com>",
        "to": [email],
        "subject": subject,
        "html": body,
    }

    email = resend.Emails.send(params)
    print(email)

