# ChatGPT-like Frontend

This project is a React-based frontend designed to mimic the ChatGPT interface.

## Tech Stack

- **Framework:** React.js
- **UI Library:** Tailwind CSS
- **Markdown Renderer:** react-markdown

## Getting Started

### Prerequisites

- Node.js (v14 or later recommended)
- npm or yarn

### Installation

1.  **Clone the repository (or navigate to this directory if already cloned):**

    ```bash
    # If you haven't cloned the main project yet
    # git clone <repository-url>
    cd <path-to-project>/frontend
    ```

2.  **Install dependencies:**

    Using npm:
    ```bash
    npm install
    ```

    Or using yarn:
    ```bash
    yarn install
    ```

### Running the Development Server

1.  **Start the development server:**

    Using npm:
    ```bash
    npm start
    ```

    Or using yarn:
    ```bash
    yarn start
    ```

    This will typically open the application in your default web browser at `http://localhost:3000`.

### Building for Production

To create a production build of the application:

Using npm:
```bash
npm run build
```

Or using yarn:
```bash
yarn build
```

This will create an optimized build in the `build` folder.

## Project Structure

```
frontend/
├── public/
│   ├── index.html         # Main HTML file
│   └── ...                # Other static assets
├── src/
│   ├── components/        # Reusable UI components
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Utility functions
│   ├── App.js             # Main application component
│   ├── index.css          # Global styles and Tailwind directives
│   ├── index.js           # React application entry point
│   └── ...                # Other source files
├── .gitignore             # Files and directories to ignore in Git
├── package.json           # Project metadata and dependencies
├── postcss.config.js      # PostCSS configuration (for Tailwind)
├── tailwind.config.js     # Tailwind CSS configuration
└── README.md              # This file
```

## Next Steps

- Implement the chat interface components.
- Set up API communication for sending messages and receiving streamed responses.
- Add features like chat history, user avatars, markdown rendering, etc. 