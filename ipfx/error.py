import logging
import functools

class FeatureError(Exception):
    """Generic Python-exception-derived object raised by feature 
    detection functions.
    """
    
def record_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result, errors = fn(*args, **kwargs)
        failed = (errors is not None)
        return (
            result,
            {'failed_fx': failed,
             'fail_fx_message': str(errors) if failed else None}
            )
    return wrapper


def fallback_on_error(fallback_value=None, 
                      suppress_errors=(FeatureError, IndexError),
                      log_errors=(Exception),
                      return_error=True):
    def fallback_on_error_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                error = None
            except suppress_errors as e:
                error = e
                result = fallback_value
                logging.warning(e)
            except log_errors as e:
                error = e
                result = fallback_value
                logging.exception("Unexpected processing error.")
            if return_error:
                return result, error
            else:
                return result
        return wrapper
    return fallback_on_error_decorator

