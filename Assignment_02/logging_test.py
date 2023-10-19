import logging
 
def main() -> None:
    # You can control the level of the logging displayed. debug is the lowest level and critical is the highest level
    logging.basicConfig(level=logging.ERROR)

    
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")


if __name__ == "__main__":
    main()