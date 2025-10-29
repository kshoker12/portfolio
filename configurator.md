# Signify Ledalite 3D Configurator

A production-grade 3D visualization and configuration system developed during my co-op at **Signify (Ledalite)**.  
This was a **critical project** and a core component of **[LightXpress 5](https://lx.ledalite.com/)**, the next-generation platform used by **5,000+ lighting agents** across North America.  

The Configurator directly **enabled the launch of [TruGroove Micro](https://www.signify.com/en-us/products/indoor-luminaires/architectural-linear/recessed/trugroove-recessed-micro?utm_source=chatgpt.com) Micro** Lighting family — an advanced architectural lighting series featuring a continuous 1.75″ aperture ribbon of light across multiple mounting styles.  


![Demo – Feature Showcase](/visualizerdemoalt.mp4)

---

## 🏗️ Legacy System

The previous generation of LightXpress relied on a **text-based configurator** where users manually entered parameters such as product codes, housing lengths, and optics.  
While functional, it lacked real-time validation and any visual feedback, leading to inefficiencies and frequent configuration mismatches which significantly slowed down configuration time for agents as the process required iterative back-and-forth collaborations between drawing engineers and agents to confirm the design.

> The process was error-prone, slow, and unintuitive — even simple layout adjustments required manual recalculation and cross-referencing.

**Legacy Architecture:** Below is an example of a pattern being configured in the previous version of **LightXpress 4**. It's evidently difficult to map out how this would look visually.

![Demo – Feature Showcase](/oldpattern.png)

---

## ⚙️ New Architecture

The new Configurator was **rebuilt from the ground up** with a focus on scalability, visualization accuracy, and seamless integration into the LightXpress ecosystem.

| Layer | Technologies | Purpose |
|:--|:--|:--|
| **Frontend Rendering Engine** | React, TypeScript, Three.js, React-Three-Fiber, TailwindCSS | Handles real-time 3D rendering, material control, and interactive module updates. |
| **State Management** | Redux Toolkit, Context API | Keeps UI, 3D scene, and backend configuration logic perfectly synchronized. |
| **Backend** | Laravel, PostgreSQL, REST APIs | Stores product schemas, module hierarchies, and configuration metadata. |
| **Deployment Pipeline** | Docker, Git CI/CD | Supports versioned product updates, modular builds, and efficient asset caching. |

This modular data-driven design allows new products to be introduced by defining JSON schema and mesh assets — requiring **no additional engineering work** on the visualization layer, resulting in an easily scalable architecture.

---

## 🧩 Module Configurator (Legacy)

In the legacy system, module configuration within a fixture was handled largely as *separate, discrete tasks*. Agents would click through a series of dropdowns or text-fields for each module segment (e.g., housing, optics, finish, sensors, louvers), with no visual context of how the modules relate to one another or to the overall fixture. Each module often had to be configured individually and repeated in adjacent positions — making multi-module fixtures inefficient and prone to ambiguity.

**Key issues:**  
- No visual interface to show module placement, orientation, light throw direction, louvers or sensor attachments.  
- No ability to select multiple modules at once or configure them collectively — each segment required its own series of clicks.  
- Ambiguities in layout and alignment: e.g., how sensors or louvers transition between modules, how light throw overlaps, etc.  
- High repetition for multi-module fixtures and inefficient workflow resulting in slower configuration and higher manual error risk.

**Legacy Module Configurator:** Below is an example of a fixture being configured in old module configurator in **LightXpress 4**

![Demo – Feature Showcase](/oldwiring.png)

---

## 🌐 New Module Configurator

The redesigned Module Configurator introduces a **visual, interactive interface** where modules within a fixture can be selected *together or independently*, configured *in bulk or singly*, and visualized via a 2D/3D hybrid interface. It resolves the ambiguities of the legacy workflow by enabling multiple-module selection and giving visual context for module interactions (light throw directions, louvers, sensors, and other attachments) that previously could not be seen.

**Key features:**  
- **Multi-select capability**: Users can select multiple modules within a fixture and apply configuration changes to all at once, reducing clicks and manual repetition.  
- **Visual feedback of modules and interactions**: The interface displays module placement, orientation, louvers, sensors, and light throw direction so that layout implications are clear in real time.  
- **2D/3D hybrid visualization**: Although the overall system uses 3D rendering, the module configurator presents a clear 2D interface for module logic, overlaid with visual cues for module interactions and dependencies.  
- **Ambiguity reduction**: Because module attributes like sensor placement, louver direction, and light throw are visually represented, users are enabled to express complex configurations in a seamless manner.
- **Streamlined workflow**: With the ability to configure modules together, apply properties in bulk, and verify visually, what used to require dozens of clicks per fixture can now be done in only a handful, with much greater confidence.

**Module Configurator in LightXpress 5**: Below is the latest module configurator which demonstrates seamless module configuration with visual feedback of light throw, louvers, and sensors. 

![Demo – Feature Showcase](/modconfiguratordemo.mp4)

**Resolving ambiguities**: Below is an example of module configurator being used to resolve ambiguities in fixture type (orientation of the T-shaped fixture).

![Demo – Feature Showcase](/modconfiguratordemo2.mp4)

---

## 🧠 Problems Solved

- **React performance optimizations** for handling **500+ active meshes** in real-time without frame drops using `use-memo` and `use-callback` hooks.
- **Unified system synchronization** across backend, Redux state, and 3D scene ensuring zero desyncs between UI and rendering layers.  
- **1:1 geometric accuracy** achieved by reconstructing complex shapes in Three.js to match engineering specs.  
- **Dynamic error detection** for invalid configurations, spacing errors, and overlapping components, ensuring legal configurations
- **Automated lighting pattern generation** enabling rapid configuration creation for common design patterns using minimal clicks.
- **Data-driven modularity** — configuration logic stored in JSON schema, allowing new products to be integrated instantly, resulting in a highly scalable system. 
- **Progressive loading and optimization** for large product assemblies, reducing load times and improving responsiveness.

---

## 🧾 Final Thoughts & Outcomes

This project marked a major milestone — both for Ledalite and for my own development as a software engineer.  
Over the course of four co-op terms, I contributed **25,000+ lines of production code**, collaborating across engineering and design teams to deliver one of the company’s most significant digital transformations.

Through this experience, I gained deep expertise in:
- Large-scale front-end architecture and state management.  
- Translating engineering specs into optimized 3D models.  
- Designing fault-tolerant full-stack systems under real production constraints.  
- Managing complexity across multiple abstraction layers — from rendering logic to backend schema design.  

The Configurator became a **cornerstone of LightXpress 5**, empowering teams to configure lighting products visually with unmatched precision.  
For me, it was a project that **transformed how I approach engineering problems** — emphasizing scalability, maintainability, and the intersection between design and technology.

---

### 💬 Summary

The **Signify Ledalite 3D Configurator** stands as a showcase of combining 3D graphics, full-stack architecture, and user-centered design to modernize enterprise visualization systems — setting a new internal standard for digital configuration at Signify Ledalite.

---