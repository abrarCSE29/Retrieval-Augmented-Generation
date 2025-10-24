

def create_api_response(status : str, message : str, **kwargs):
    response =  {
        "status" : status,
        "message" : message,
    }
    response.update(kwargs)

    return response