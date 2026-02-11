import sys

def error_message_detail(error, error_detail):

    # If error_detail is the sys module, get exc_info()
    if error_detail == sys:
        _, _, exc_tb = sys.exc_info()
    else:
        # Otherwise assume it's the exc_info tuple
        _, _, exc_tb = error_detail

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    else:
        # Fallback if no traceback information is available
        error_message = f"Error occurred: {str(error)}"
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
