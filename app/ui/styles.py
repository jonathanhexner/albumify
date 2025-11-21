"""CSS styles for the application."""


def get_custom_css() -> str:
    """Get custom CSS for the application."""
    return """
    <style>
        /* Main container */
        .main {
            padding: 1rem;
        }

        /* Chat messages */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }

        /* Status indicators */
        .status-ready {
            color: #4caf50;
            font-weight: 600;
        }

        .status-error {
            color: #f44336;
            font-weight: 600;
        }

        .status-warning {
            color: #ff9800;
            font-weight: 600;
        }

        /* Workflow steps */
        .workflow-step {
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 0.25rem;
            font-family: monospace;
            font-size: 0.9rem;
        }

        .step-completed {
            background-color: #c8e6c9;
            color: #2e7d32;
        }

        .step-running {
            background-color: #fff9c4;
            color: #f57f17;
        }

        .step-pending {
            background-color: #eeeeee;
            color: #757575;
        }

        .step-failed {
            background-color: #ffcdd2;
            color: #c62828;
        }

        /* Sidebar sections */
        .sidebar-section {
            margin-bottom: 2rem;
        }

        /* Tool cards */
        .tool-card {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background-color: #fafafa;
            border-radius: 0.25rem;
            border-left: 3px solid #2196f3;
        }

        /* Info boxes */
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }

        /* Example query buttons */
        .example-query {
            margin: 0.25rem;
        }
    </style>
    """
