# script.py
# -*- coding: utf-8 -*-
import logging
import dlib
import numpy as np
from skimage import io
from scipy.spatial import distance
from PIL import Image, ImageDraw
import tempfile


# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация моделей
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("./datasets/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(
    "./datasets/dlib_face_recognition_resnet_model_v1.dat"
)


def save_temp_image(image):
    """Сохраняет изображение во временный файл и возвращает путь"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, "JPEG")
            return tmp.name
    except Exception as e:
        logger.error(f"Ошибка сохранения временного файла: {e}")
        return None


def annotate_face(image):
    """Добавляет аннотации к изображению лица"""
    try:
        if image is None:
            return None

        img = io.imread(image) if isinstance(image, str) else image
        dets = detector(img, 1)

        if len(dets) != 1:
            return None

        shape = sp(img, dets[0])
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Рисуем точки
        for i in range(0, 68):
            x, y = shape.part(i).x, shape.part(i).y
            draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))

        return save_temp_image(pil_img)
    except Exception as e:
        logger.error(f"Ошибка аннотирования: {e}")
        return None


def face_comparator(img1_url, img2_url):
    """Сравнивает два лица и возвращает результат"""
    try:
        logger.info("Начало обработки изображений")

        # Обработка первого изображения
        img1 = io.imread(img1_url)
        dets1 = detector(img1, 1)
        if len(dets1) != 1:
            logger.warning("На первом изображении не найдено лицо")
            return None, None, None, None

        shape1 = sp(img1, dets1[0])
        desc1 = facerec.compute_face_descriptor(img1, shape1)
        annotated1 = annotate_face(img1)

        # Обработка второго изображения
        img2 = io.imread(img2_url)
        dets2 = detector(img2, 1)
        if len(dets2) != 1:
            logger.warning("На втором изображении не найдено лицо")
            return None, None, None, None

        shape2 = sp(img2, dets2[0])
        desc2 = facerec.compute_face_descriptor(img2, shape2)
        annotated2 = annotate_face(img2)

        # Сравнение
        dist = distance.euclidean(desc1, desc2)
        result = "Это один и тот же человек" if dist < 0.6 else "Это разные люди"

        return result, dist, annotated1, annotated2

    except Exception as e:
        logger.error(f"Ошибка сравнения: {e}")
        return None, None, None, None
    finally:
        # Очистка временных файлов
        pass
