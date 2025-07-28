Here’s a revised version of your write-up that emphasizes the mimicry of Cursor's background agent, the use of **three distinct agents**, and the **NeMo toolkit** foundation:

---

# Blake's Hackathon Version of the AIQ Toolkit

This is Blake’s experimental hackathon version of the AIQ toolkit. While documentation is minimal due to time constraints, the project is a **lightweight background coding assistant** inspired by tools like **Cursor’s background agent**, but built using NVIDIA’s **NeMo Agent Toolkit**.

It’s designed as a **proof-of-concept** that mimics the way Cursor continuously interprets and reacts to code changes in the background. However, instead of a single monolithic agent, this version uses **three modular agents** that work together:

- 🔍 **Intent Parsing Agent** – understands what the user wants.
- 🛠 **File Planning Agent** – determines which parts of the code to modify.
- 🧠 **Code Modification Agent** – generates the actual code updates.

It’s early-stage and may feel **a bit slow or rough around the edges**, but the architecture lays the groundwork for a more scalable and intelligent background dev assistant.

---

## 📁 Code Location

Hackathon code is located in:

```
examples/basic/frameworks/background_agents
```

---

## 🧠 How It Works

<a href="https://ibb.co/LD8wFyY1">
  <img src="https://i.ibb.co/FkKvr1mz/Screenshot-2025-07-27-at-6-31-57-PM.png" alt="AIQ Toolkit Screenshot" border="0">
</a>

---

**Example of first PR**
[https://github.com/blaketylerfullerton/NeMo-Agent-Toolkit-UI/pull/1](https://github.com/blaketylerfullerton/NeMo-Agent-Toolkit-UI/pull/1) <img width="1085" height="516" alt="Screenshot 2025-07-27 at 6 38 27 PM" src="https://github.com/user-attachments/assets/a1db2a92-cf2a-48b2-b989-28fed85cf4c9" />
