from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

from src.user import add_user

app = FastAPI()
templates = Jinja2Templates(directory='templates/')

@app.get('/forecast')
def form_post(request: Request):
    result = 'Type your phone number'
    return templates.TemplateResponse('forecast.html', context={'request': request, 'result': result})

@app.post('/forecast')
def form_post(request: Request, number: int = Form(...),carrier: str = Form(...), enter_nyc: int = Form(...), leave_nyc: int = Form(...)):
    result = add_user(number,carrier,enter_nyc,leave_nyc)
    return templates.TemplateResponse('forecast.html', context={'request': request, 'result': result})