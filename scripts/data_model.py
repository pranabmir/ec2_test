from pydantic import BaseModel
from pydantic import EmailStr,HttpUrl

class NLPDataInput(BaseModel):
    text: list[str]
    user_id: EmailStr
    
class ImageDataInput(BaseModel):
    url: list[HttpUrl]
    user_id: EmailStr
    
class NLPDataOutput(BaseModel):
    mod_name: str
    text: list[str]
    targets:list[str]
    scores:list[float]
    prediction_time: float
class ImageDataOutput(BaseModel):
    mod_name: str
    url: list[HttpUrl]
    targets: list[str]
    scores: list[float]
    prediction_time: float