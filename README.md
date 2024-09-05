# rentahal
RENT-A-HAL Universal LLM Broker v1.1 (OneRing) https://rentahal.com
# RENT-A-HAL: Revolutionizing On-Prem AI with the **1.1 (OneRing)** Interface

Welcome to **RENT-A-HAL**, the groundbreaking platform designed to democratize access to AI by bringing powerful, scalable, on-prem AI solutions to everyone at an incredibly low cost. With our cutting-edge **1.1 (OneRing)** interface, users can seamlessly interact with AI models across various tasks, all while enjoying the benefits of a fully decentralized, plug-and-play AI infrastructure.

> **Public Demo**: Experience the future of AI at [RENT-A-HAL Public Demo](http://rentahal.com)

## Table of Contents

- [Vision and Mission](#vision-and-mission)
- [Key Features](#key-features)
- [Installation](#installation)
- [Use Cases](#use-cases)
- [AI Worker Nodes & Scaling](#ai-worker-nodes--scaling)
- [Current State of Development](#current-state-of-development)
- [Subsystems](#subsystems)
- [Roadmap](#roadmap)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contributors](#contributors)

## Vision and Mission

At **The N2NHU Lab for Applied AI**, we are driven by the vision of making AI accessible to everyone. By leveraging consumer-grade hardware and decentralized AI worker nodes, we aim to disrupt traditional cloud-based AI models and offer users unprecedented control and scalability. Our **mission** is to create the world’s most powerful distributed AI cloud through an intuitive, easy-to-use system that scales to meet the needs of individuals and enterprises alike.

With **RENT-A-HAL**, you no longer need complex setups or expensive cloud-based solutions. Instead, you can run a high-powered AI network from your own hardware, joining the ranks of those building the future of AI.

## Key Features

- **Impressive 1.1 (OneRing) Interface**: Our **1.1 (OneRing)** interface is designed for **simplicity** and **power**, enabling users to access AI tasks effortlessly. From chat to vision and image generation, every feature is accessible through this sleek, intuitive UI.
- **Brave Public Demo**: Join the revolution at [RENT-A-HAL Public Demo](http://rentahal.com) and experience the future of AI interaction in real time.
- **Decentralized AI Cloud**: By joining the **RENT-A-HAL** public network, users can install worker nodes on their systems and gain unlimited access to AI services via the public web interface.
- **On-Prem AI at Minimal Cost**: Set up powerful AI solutions with hardware costs as low as $1500 and minimal power costs (~$200/month), bringing enterprise-level AI within reach of everyone.
- **Universal Real-Time Brokering**: The platform queues and distributes AI tasks (chat, vision, imagine) across worker nodes for real-time processing, ensuring performance and scalability.
- **Plug-and-Play AI Worker Nodes**: Easily set up AI worker nodes on Nvidia 30xx GPUs and integrate them into the public network via Ngrok for fast, scalable AI processing.
- **Sysop Features for Control**: System administrators can manage workers, monitor their health, and blacklist underperforming nodes through the sysop panel.
- **Claude API Integration**: Seamlessly integrate with **Claude's API** for high-quality language model processing and real-time feedback.

## Installation

### Prerequisites

- Python 3.x
- Node.js
- Nvidia GPU (e.g., 30xx series)
- Ngrok for public-facing endpoints
- Claude API key

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/rent-a-hal.git
   cd rent-a-hal
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up your worker node (if applicable):

Run the AI worker node setup on your machine and expose it with Ngrok:

bash
Copy code
ngrok http 5000
Run the backend server:

bash
Copy code
python webgui.py
Access the web interface:

Visit http://localhost:5000 to start using RENT-A-HAL.

Use Cases

Individuals:
Build your own AI infrastructure at home for personal projects, research, or entertainment.

Enterprises:
Deploy AI worker nodes across your organization and scale effortlessly without the need for expensive cloud solutions.

Public Network Users: 
Install AI worker nodes on your system and gain unlimited access to the RENT-A-HAL web interface by contributing your compute power to the public network.

AI Worker Nodes & Scaling

Universal Real-Time Brokering

RENT-A-HAL brokers AI tasks in real time, distributing them across worker nodes based on demand. Whether it’s handling chat queries, image analysis, or generating complex visual outputs, the platform ensures that tasks are efficiently managed and queued for optimal performance.

Plug-and-Play AI Worker Nodes

All AI tasks are executed by worker nodes. This plug-and-play model allows for easy scaling—simply add more worker nodes to the network to increase capacity and throughput. It’s the perfect solution for both small setups (e.g., three nodes) and large-scale deployments.

Low Cost, High Efficiency:
A setup of three worker nodes, costing around $1500, can run on standard 110v power with a monthly cost of approximately $200. This setup brings enterprise-level AI infrastructure into the hands of anyone willing to contribute their compute power.

Scaling: 
Whether you’re running three nodes or hundreds, RENT-A-HAL scales effortlessly, distributing workloads across all available nodes.

Current State of Development

Fully Functional Features

Real-Time Query Processing: 
Supports chat, vision, and image generation tasks, with task queuing and real-time feedback.

Worker Node Integration:
Fully functional AI worker nodes can be deployed with Claude API support for chat and task processing.

Sysop Panel:
Admins have control over system stats, worker management, health monitoring, and blacklist management.

Public Demo:
The brave public demo of RENT-A-HAL is live at RENT-A-HAL Public Demo, offering users a glimpse into the platform's capabilities.

In Development

Public Worker Node Onboarding: We're streamlining the process for public users to join the RENT-A-HAL network by installing worker nodes via an easy setup process.

Health Monitoring Enhancements: Expanding on the health restoration routine to improve the reintegration of previously blacklisted workers.

Subsystems

Sysop Features

System Statistics:
Track the total number of queries, processing times, and costs in real-time.

Worker Health Monitoring:
Actively monitors worker node performance. If a node’s health drops below a set threshold, it is blacklisted until its health is restored.

Blacklist & Recovery: Workers that are blacklisted can be automatically reintegrated into the system once their health improves, ensuring high performance.

AI Worker Management

Sysops can add, remove, and manage AI worker nodes. Worker nodes can be assigned specific tasks (chat, vision, imagine) and are monitored for health and performance to ensure the system runs smoothly.

Universal Real-Time Task Brokering
The universal task brokering system queues and processes tasks in real-time, allowing for seamless scaling and efficient task management across a decentralized AI cloud.

Roadmap

Public Worker Node Integration:
Simplifying the process for public users to join the RENT-A-HAL network by installing their own worker nodes via ngrok.

Advanced Health Monitoring:

Adding more granular health monitoring and predictive analytics to prevent node failures.

Global Expansion: Expanding the public network to create the largest decentralized AI cloud powered by consumer hardware.

Technologies Used

Python (Flask): Backend server.

JavaScript: Frontend logic for real-time interactions.

Tailwind CSS: Responsive UI styling.

WebSocket: For real-time communication between frontend and backend.

Ngrok: Exposes public worker nodes for global access.

Claude API: Powers task processing for chat and AI interaction.

License

Work product of The N2NHU Lab for Applied AI in Newburgh NY USA. No source is published.

Contributors

Designer: N2NHU

Architect: HAL (gpt4o)

Coder: Claude (Sonnet 3.5)
