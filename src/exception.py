import logging
import sys 


def error_message_detail(err,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    error_message = "Error occured in python script name [{0}] at line number [{1}] error message [{2}] ".format(exc_tb.tb_frame.f_code.co_filename,exc_tb.tb_lineno,str(err))
    return error_message

class CustomException(Exception):
    def __init__(self, err, error_detail:sys):
        super().__init__(err)
        self.error_message = error_message_detail(err,error_detail=error_detail)
    
    