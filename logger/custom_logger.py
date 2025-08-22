import os
import logging
from datetime import datetime
import structlog

class CustomLogger:
    # Class-level flag: ensures we configure logging only once per run
    _configured = False

    def __init__(self,log_dir="logs",log_file=None):
        """
        Constructor (runs when you create a CustomLogger object).
        Sets up where log files should live and what file name to use.
        """
        # Ensure logs directory exists
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Local variable (temporary) for log_file if not supplied
        # Timestamped log file (for persistence)
        if log_file is None:
            log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        
        #log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

        self.log_file_path = os.path.join(self.logs_dir, log_file)


    def get_logger(self, name=__file__):
        """
        Returns a structlog logger.
        Configures logging only once per process (avoids multiple empty files).
        """
        # Local variable: just the "basename" of the module asking for a logger
        logger_name = os.path.basename(name)

        # Configure logging for console + file (both JSON) and only configure once
        if not CustomLogger._configured:
            # Local variable: file handler, tied to *this* object's log file
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON lines

        # Local variable: console handler, for printing logs to terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure Logging
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(message)s",   # Structlog will handle JSON rendering
        #     handlers=[console_handler, file_handler]
        # )

        # replaced Configure logging code above
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        if not root.handlers:
            root.addHandler(console_handler)
            root.addHandler(file_handler)


        # Configure structlog for JSON structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        # Flip the flag so we don’t configure again on future calls
        CustomLogger._configured = True

        # Return a structlog logger bound to this module’s name
        return structlog.get_logger(logger_name)


# --- Usage Example ---
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)  
    logger.info("User uploaded a file", user_id=123, filename="report.pdf")
    logger.error("Failed to process PDF", error="File not found", user_id=123)
        