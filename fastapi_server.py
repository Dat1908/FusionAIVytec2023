import os
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from typing import Optional

# Import các hàm xử lý
from body_shape_calculator import get_body_shape
from face_shape_detector import load_face_model, get_face_shape
from skin_hair_color_detector import get_skin_color, get_hair_color, personal_color

# Khởi tạo FastAPI
app = FastAPI()

# Thiết lập đường dẫn tới thư mục templates
templates = Jinja2Templates(directory="templates")
app.mount(".\static", StaticFiles(directory="static"), name="static")

# Face shape classes
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
model = load_face_model()
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/face_shape")
async def face_shape_func(request: Request, file: UploadFile = File(...)):
    try:
        path_to_save = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(path_to_save, "wb") as f:
            f.write(file.file.read())

        if not detect_face(path_to_save):
            return templates.TemplateResponse("face_shape.html", {"request": request, "msg": "Ảnh không hợp lệ"})

        face_shape = get_face_shape(model, classes, image_path=path_to_save)
        label = f'Face: {face_shape}\n'
        return templates.TemplateResponse("face_shape.html", {"request": request, "label": label, "msg": "Tải file lên thành công"})
    except Exception as ex:
        print(ex)
        return templates.TemplateResponse("face_shape.html", {"request": request, "msg": "Không nhận diện được vật thể"})

@app.post("/body_shape")
async def body_shape_func(request: Request, Bust: int = Form(...), Waist: int = Form(...), Hip: int = Form(...)):
    body_shape = get_body_shape(Bust, Waist, Hip)
    return templates.TemplateResponse("body_shape.html", {"request": request, "body_shape": body_shape, "msg": "Thành công!"})

@app.post("/personal_color")
async def personal_color_func(request: Request, file: UploadFile = File(...)):
    try:
        path_to_save = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(path_to_save, "wb") as f:
            f.write(file.file.read())

        if not detect_face(path_to_save):
            return templates.TemplateResponse("personal_color.html", {"request": request, "msg": "Ảnh không hợp lệ"})

        skin_color = get_skin_color(path_to_save)
        hair_color = get_hair_color(path_to_save)
        label = personal_color(skin_color, hair_color)
        return templates.TemplateResponse("personal_color.html", {"request": request, "label": label, "msg": "Tải file lên thành công"})
    except Exception as ex:
        print(ex)
        return templates.TemplateResponse("personal_color.html", {"request": request, "msg": "Không nhận diện được vật thể"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
