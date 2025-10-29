
# **Compass App — Personal Note**

*Compass App is a system I personally use every day for the past 3 years. Building it fundamentally changed how I approach productivity, discipline, and goal‑setting — it gave me structure across my career, training, and personal growth. It’s the framework that helped me stay focused, consistent, and intentional about my long‑term goals.*  

*All data showcased in this documentation is **demo data** created for presentation purposes only. None of it reflects real personal or financial information.*

---


# **Chapter 1 — Feature Overview**

Compass App is a full-stack **life‑management ecosystem** that unifies productivity, goal‑tracking, fitness, nutrition, and self‑development into one intelligent mobile experience.  
Built with **React Native + Redux Toolkit** on the frontend and a **Django REST Framework + PostgreSQL** backend, it combines modern engineering with practical life‑organization workflows.  

This chapter highlights the **core value and user experience** of each feature while briefly summarizing its underlying technology.  
For detailed engineering insights, see [Chapter 2 — Technical Architecture & Implementation →](#chapter-2--technical-architecture--implementation).

---

## 🧭 **Archetypes (System of Character Development)**

**Purpose:**  
A psychological growth engine that helps users understand and evolve their behavior through **three core archetypes** — *The Dedicated Learner*, *The Disciplined Engineer*, and *The Visionary Builder*.  
Each archetype defines its own set of **principles**, **examples**, and **daily quests**, creating a gamified loop of reflection → action → improvement.

**Key Features:**
- **Rule System:** Each archetype has a “Main Deck” and “Side Deck” of personal rules.  
- **Leveling Mechanism:** XP earned through daily quests and ratings; archetypes level up dynamically.  
- **Examples & Reflection:** Users attach real‑world examples of applying each rule.  
- **Credits System:** Earn credits for progress and quest completion.  
- **Visual Card Design:** Every rule and archetype is rendered as an interactive card for intuitive navigation.

**Frontend ↔ Backend Integration:**  
- The frontend stores archetype state in a dedicated `archetypeSlice`, handling deck management and XP progression.  
- The backend (`Archetype`, `RuleV2`, `ExampleV2`, `Level`) models the psychology framework, with APIs for rule updates, example logging, and XP distribution.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/archetype.mp4)

---

## 🧱 **Responsibility Manager (Productivity & Task System)**

**Purpose:**  
Acts as the user’s **daily command center**, structuring life responsibilities into actionable tasks and subtasks.  
The Responsibility Manager transforms routine productivity into a reward‑driven workflow powered by AI automation.

**Key Features:**
- **Hierarchical Structure:** Responsibilities → Tasks → Subtasks.  
- **AI PDF Task Generation:** Upload syllabus or documents; backend uses OpenAI to generate structured subtasks automatically.  
- **Drag‑and‑Drop Todo List:** Intuitive task reordering with real‑time Redux updates.  
- **Hour & Progress Tracking:** Quantify work time and see completion percentages per category.  
- **Credits Reward:** Earn credits for task completion and daily consistency.

**Frontend ↔ Backend Integration:**  
- Frontend implements a `DraggableFlatList` for reordering and a `beastmodeSlice` for state management.  
- Backend manages `Responsibility`, `Task`, and `SubTask` models with REST endpoints and AI PDF processing via `generate_subtasks_from_pdf`.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/responsibility1.mp4)

---

## 🎯 **Compass (Goal‑Tracking System)**

**Purpose:**  
A structured **goal hierarchy** that aligns long‑term ambitions with short‑term execution.  
Users define **Yearly**, **Monthly**, and **Weekly Compasses** — each with destinations, points, and mistake tracking for accountability.

**Key Features:**
- **Three‑Tier Goal Structure:** Year → Month → Week.  
- **Destination Cards:** Visual representation of objectives and progress.  
- **Point System:** Quantifies achievement (1–10 scale) and percentage completion.  
- **AI Sentiment Analysis:** Weekly destination points are influenced by AI sentiment scoring of written reflections, ensuring unbiased and consistent self‑evaluation.  
- **Mistake Tracking:** Negative points for missed goals to reinforce discipline.  
- **Credits Integration:** Earning credits on goal completion.

**Frontend ↔ Backend Integration:**  
- Frontend `compassSlice` manages hierarchical data with memoized selectors and progress calculations.  
- Backend models (`Compass`, `YearlyCompass`, `MonthlyCompass`, `WeeklyCompass`, `Destination`) compute point ratios and weekly percentages.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/compass.mp4)

---

## 🥊 **Training (Sports & Performance Analytics)**

**Purpose:**  
Transforms boxing and fitness tracking into a data‑driven performance dashboard.  
Built for athletes and enthusiasts to log sessions, analyze techniques, and record progress visually.

**Key Features:**
- **Boxing Profile:** Personal record of fights, stats, and attributes.  
- **Session Tracking:** Round‑by‑round performance logging.  
- **Nuggets System:** Knowledge capture for tactics and insights.  
- **Fight Analysis:** Break down techniques and outcomes.  
- **Focus Areas:** Identify strengths and weaknesses.  

**Frontend ↔ Backend Integration:**  
- Frontend `trainingSlice` connects to a comprehensive API covering `BoxingSession`, `SparringSession`, and `FocusArea`.  
- Backend leverages related models for sessions, techniques, and statistics aggregation.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/training.mp4)

---

## 🍎 **Diet (Nutrition & Health Tracking)**

**Purpose:**  
A smart nutrition system to log meals, track macros, and visualize progress toward dietary goals.  
Integrates seamlessly with Training and Dashboard modules for a holistic health view.

**Key Features:**
- **Meal Logging & Nutrient Cards:** Track calories and macros in real time.  
- **Activity Integration:** Sync with training sessions for net calorie balance.  
- **Recipe Creation:** Build custom meals and calculate macros.  
- **Visualization:** Progress graphs and macro distribution charts.  

**Frontend ↔ Backend Integration:**  
- `dietSlice` and `dietApi` handle nutrition data and UI rendering.  
- Backend models (`Meal`, `Activity`, `Nutrient`, `Recipe`) power daily tracker logic and macro aggregation.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/diet.mp4)

---


---

## 💰 **Finance (Expense, Income, and Net Worth Tracking)**

**Purpose:**  
Provides a unified and automated overview of the user’s financial health by combining expense tracking, savings, income, and investments into one module.  
This system integrates seamlessly with iOS Shortcuts for real‑time transaction ingestion and classification, eliminating the need for manual entry.

**Key Features:**
- **Automated Transactions:** Generated and classified automatically through iOS Shortcuts calling a secure backend API.  
- **Expense Categorization:** Merchant normalization and rule‑based tagging (Food, Transport, Subscriptions, etc.).  
- **Net Worth Dashboard:** Aggregates data from debit cards, savings, assets, and investments.  
- **Credit & Debit Management:** Real‑time tracking of available balances, limits, and payments.  
- **Income & Paycheck Tracking:** Automatically logs new income entries and adjusts net worth.  
- **Investment Overview:** Displays stocks, savings targets, and asset valuation over time.  

**Frontend ↔ Backend Integration:**  
- Frontend `financesSlice` manages ledgers, categories, and asset tracking through a unified state model.  
- Backend includes `Transaction`, `Expense`, `CreditCard`, `DebitCard`, `Asset`, `Saving`, `Paycheque`, and `Stock` models for a complete financial schema.  
- iOS Shortcuts call `/api/finance/transactions/ingest/`, which auto‑classifies transactions and updates expense categories, balances, and net worth metrics in real time.  

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/finances.mp4)

---

## 🏠 **Dashboard (Home & Analytics Hub)**

**Purpose:**  
The central command hub that visualizes user progress across all modules.  
Combines data from productivity, goals, training, **finances**, and diet to present a cohesive snapshot of personal growth.

**Key Features:**
- **Unified Statistics:** Credits, Compass %, Hours Tracked, Weight, and Net Worth.  
- **Automated Transactions:** **Transactions are automatically generated and classified** via **iOS Shortcuts** that call a secured backend API — enabling near real‑time expense tracking without manual entry.  
- **Charts & Visuals:** Progress and hours breakdowns with real‑time Redux updates.  
- **Quick Navigation:** Access to all modules through interactive cards.  
- **Dynamic Themes:** User‑selectable color schemes with persistent storage.  
- **Quote of the Day:** Motivational integration for daily focus.  

**Frontend ↔ Backend Integration:**  
- `homeSlice` and modular selectors aggregate stats from all feature slices.  
- Backend exposes read‑only summary endpoints (credits, hours, net worth) and **transaction ingestion endpoints** used by iOS Shortcuts for classification and ledger updates.

*→ [See Technical Details →](#chapter-2--technical-architecture--implementation)*  

![Demo – Feature Showcase](/video/home.mp4)

---
# **Chapter 2 — Technical Architecture & Implementation**

This chapter details the **engineering architecture** behind Compass App — focusing on how React Native, Redux Toolkit, and Django REST Framework come together to power the features introduced in Chapter 1.  
It outlines the system design, state management, API integration, AI automation, and performance strategies that make the app robust, scalable, and production‑ready.

---

## 🏗️ **Architecture Overview**

Compass App is built with a **modern, modular architecture** using:
- **Frontend:** React Native (Expo) + TypeScript + Redux Toolkit
- **Backend:** Django REST Framework (DRF) + PostgreSQL
- **AI Integration:** OpenAI API for PDF‑based task generation
- **Security:** JWT authentication + secure local storage

**Frontend & Backend Interaction**
1. The React Native app uses Redux slices and async thunks to communicate with the Django REST API.
2. All modules (Responsibility Manager, Compass, Diet, Training, Archetypes, Dashboard) have dedicated slices for state management.
3. The backend exposes RESTful endpoints for each model, handling CRUD operations, analytics, and AI‑powered automation.

```text
React Native (Expo) + Redux Toolkit
          ↓  (Axios API calls)
Django REST Framework (DRF)
          ↓
PostgreSQL Database
```

---

## 🧩 **Frontend Architecture**

### **Redux Toolkit Implementation**
- Each feature module (e.g., `compassSlice`, `archetypeSlice`, `trainingSlice`, `dietSlice`) has its own slice.
- Async Thunks handle all backend communication for fetching, posting, and updating data.
- Memoized selectors and normalized state ensure high performance and minimal re‑renders.

### **Highlights:**
- **Modular Slice Pattern:** Isolated state per feature for scalability.
- **Async Thunks:** Integrated API logic for async calls.
- **Memoization:** Used with `useMemo` and `useCallback` for efficient computations.
- **TypeScript Safety:** Strongly‑typed Redux state across all modules.

```typescript
interface CompassState {
  yearlyCompasses: YearlyCompass[];
  activeCompass: WeeklyCompass | null;
  loading: boolean;
}
```

### **Key Frontend Systems**
| System | Description |
|--------|--------------|
| **AI Task Generator** | Uses `expo-document-picker` for file uploads, sends PDFs to the backend for AI parsing. |
| **Gamification Engine** | Cross‑module Credits system connected to all slices. |
| **Drag‑and‑Drop UI** | Implemented with `react-native-draggable-flatlist`. |
| **Theming Engine** | Dynamic multi‑theme color system with secure storage. |
| **Animations** | Native driver‑based animations for 60fps transitions. |

---

## ⚙️ **Backend Architecture (Django REST Framework)**

### **Core Design Principles**
- **Model‑driven architecture:** Each module corresponds to a Django model.
- **Hierarchical Relationships:** Responsibilities → Tasks → SubTasks; Yearly → Monthly → Weekly Compass.
- **Serializer Abstraction:** Data validation and transformation for each API endpoint.
- **Management Commands:** Scripted automation for seeding data and simulating realistic workloads.

### **Core Models by Module**

| Module | Key Models | Description |
|--------|-------------|-------------|
| **Archetypes** | `Archetype`, `RuleV2`, `ExampleV2`, `Level` | Tracks personality archetypes, principles, examples, and XP. |
| **Responsibility Manager** | `Responsibility`, `Task`, `SubTask` | Handles productivity tasks, subtasks, and AI task generation. |
| **Compass** | `Compass`, `YearlyCompass`, `MonthlyCompass`, `WeeklyCompass`, `Destination` | Multi‑tier goal tracking with progress and points. |
| **Training** | `BoxingSession`, `SparringSession`, `FocusArea` | Records boxing performance, sparring data, and improvement focus areas. |
| **Diet** | `Meal`, `Activity`, `Nutrient`, `Recipe`, `DailyTracker` | Logs nutrition and activity data. |
| **Finance** | `Transaction`, `Expense`, `CreditCard`, `DebitCard`, `Asset`, `Saving`, `Paycheque`, `Stock` | Unified system for tracking expenses, income, assets, and net worth. |

---

## 🧠 **Key Technical Achievements**

### 1. **AI‑Powered PDF Task Generation**
- **Frontend:** Uploads PDFs via Expo Document Picker, sends FormData to backend.
- **Backend:** Parses text using PyPDF2 → sends to OpenAI API → generates subtasks JSON → inserts into DB.
- Enables automated task creation for coursework, projects, or documents.

### 2. **Hierarchical Goal System**
- Three‑tier Compass structure: Yearly → Monthly → Weekly.
- Dynamic point system calculates progress percentages and lifetime stats.
- Mistake‑tracking introduces negative scoring for accountability.

### 3. **Gamified Credits Engine**
- Cross‑module credit rewards tied to achievements and goals.
- Real‑time Redux updates for completed actions.
- Lifetime tracking visible in Dashboard.

### 4. **Dynamic Archetype System**
- Backend XP and leveling logic synchronized with frontend UI.
- Rules, examples, and levels connected through relational models.
- Custom management commands for XP seeding and archetype setup.

### 5. **Drag‑and‑Drop Productivity UX**
- Native‑feeling drag reordering for subtasks.
- Optimistic UI updates for smooth interactivity.
- Backend syncs order persistently using custom API routes.

### 6. **iOS Shortcuts — Transaction Ingestion & Classification**
- **Flow:** An iOS Shortcut sends a signed request (with metadata like amount, merchant, and card) to a protected DRF endpoint (e.g., `POST /api/finance/transactions/ingest/`).  
- **Server:** The backend authenticates the call (JWT or signed secret), stores the transaction, then **auto‑classifies** it (rule‑based categorization + merchant normalization).  
- **Result:** The Dashboard reflects **new transactions automatically**, updating expense categories and **Net Worth** metrics without manual entry.
- **Benefits:** Real‑time expense capture, zero‑touch classification, tighter Dashboard KPIs.

**Example Payload (from iOS Shortcuts):**
```json
{
  "timestamp": "2026-01-20T17:03:12Z",
  "amount": -14.28,
  "merchant": "Starbucks #1234",
  "card_hint": "Visa-1234",
  "notes": "morning coffee",
  "source": "ios-shortcut"
}
```

**Classification Response (server):**
```json
{
  "id": 9812,
  "category": "Food & Drink",
  "normalized_merchant": "Starbucks",
  "status": "ingested",
  "ledger_effect": "posted"
}
```


### 8. **Finance Engine — Unified Ledger and Net Worth System**
- **Backend:** Combines transactions, expenses, income, and investments into a single, relational ledger.  
- **Automatic Ingestion:** iOS Shortcuts send real‑time transaction data to `/api/finance/transactions/ingest/`, triggering rule‑based classification and merchant normalization.  
- **Frontend:** The `financesSlice` aggregates all financial data into a unified dashboard view showing assets, net worth, and expenses.  
- **Analytics:** Django ORM queries calculate running balances, credit utilization, and category spending patterns.  
- **Investment & Asset Integration:** Tracks user holdings, savings goals, and valuation updates over time.  
- **Outcome:** A live, data‑driven financial insight system that ties into the Dashboard for complete personal visibility.


### 7. **Secure Authentication Layer**
- JWT‑based authentication and refresh token system.
- Secure token storage via Expo Secure Store.
- Optional biometric login configuration.

---

## ⚡ **Performance & Scalability**

| Strategy | Implementation |
|-----------|----------------|
| **Virtualized Lists** | Used FlatList + DraggableFlatList for large data sets. |
| **Memoization** | `useMemo` + `useCallback` for efficient recalculations. |
| **Batch Updates** | Grouped Redux updates to minimize renders. |
| **Async Loading States** | Global `AppLoader` and per‑slice loading flags. |
| **Lazy Screen Loading** | Screens load on demand for better startup time. |
| **Backend Caching** | Serializer caching for high‑traffic endpoints. |

---

## 🔒 **Security & Data Integrity**

- JWT Authentication for secure access control.  
- HTTPS‑only API communication.  
- Sensitive data encrypted in local storage.  
- PostgreSQL used for robust relational data handling.  
- Token refresh logic with automatic re‑authentication.

---

## 📈 **Scalability & Maintainability**

- **Feature‑based file organization** for modular development.  
- **Reusable components** and shared hooks for maintainability.  
- **Separation of concerns** across frontend slices and backend apps.  
- **TypeScript typing + Django serializers** ensure consistent contracts between frontend and backend.

**Directory Structure:**
```text
src/
  screens/          # Feature screens
  store/            # Redux state management
    slices/         # Feature slices (compass, diet, etc.)
    api/            # API integrations
    selectors/      # Memoized selectors
  components/       # Reusable components
  types/            # Type definitions
  Constants/        # App constants
  Helpers/          # Utility functions
```

---

## 🧩 **Engineering Summary**

Compass App demonstrates the combination of **user experience design and technical sophistication** through:  
- AI‑integrated productivity automation,  
- Hierarchical data management across multiple life domains,  
- Advanced state synchronization with Redux, and  
- Backend logic driven by structured Django models — **plus automated financial transaction ingestion via iOS Shortcuts**.

Together, these systems create a powerful, scalable, and maintainable application — representing an engineering solution that bridges **personal development** and **technical precision**.

---
