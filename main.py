# -*- coding: utf-8 -*-
import logging
import time
import vk_api
import numpy as np
import sys
from script import face_comparator
from VK_API_KEY import VK_API_KEY
import os

# Фикс кодировки для Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class VKBot:
    def __init__(self, token):
        self.vk = vk_api.VkApi(token=token)
        self.upload = vk_api.VkUpload(self.vk)
        try:
            self.vk._auth_token()
            logger.info("Успешная авторизация в VK API")
        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            raise

    def process_message(self, message):
        try:
            user_id = message["last_message"]["from_id"]
            attachments = message["last_message"].get("attachments", [])

            if len(attachments) == 2:
                img1_url = attachments[0]["photo"]["sizes"][-1]["url"]
                img2_url = attachments[1]["photo"]["sizes"][-1]["url"]

                logger.info(f"Получены фото от пользователя {user_id}")

                # Сравниваем лица
                result, distance, annotated_img1, annotated_img2 = face_comparator(
                    img1_url, img2_url
                )

                if result is None:
                    self.send_message(user_id, "Не удалось обработать фотографии")
                    return

                # Формируем сообщение
                message_text = (
                    f"Результат сравнения: {result}\n"
                    f"Евклидово расстояние: {distance:.4f}"
                )

                # Отправляем результат
                self.send_message(user_id, message_text)

                # Отправляем фото если есть
                if annotated_img1:
                    photo = self.upload.photo_messages(annotated_img1)[0]
                    attachment = f"photo{photo['owner_id']}_{photo['id']}"
                    self.send_message(
                        user_id, "Первое фото с опорными точками:", attachment
                    )

                if annotated_img2:
                    photo = self.upload.photo_messages(annotated_img2)[0]
                    attachment = f"photo{photo['owner_id']}_{photo['id']}"
                    self.send_message(
                        user_id, "Второе фото с опорными точками:", attachment
                    )

            else:
                self.send_message(
                    user_id,
                    "Пожалуйста, отправьте ровно 2 фотографии одним сообщением.",
                )
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")

    def send_message(self, user_id, message, attachment=None):
        try:
            params = {"random_id": 0, "peer_id": user_id, "message": message}
            if attachment:
                params["attachment"] = attachment
            self.vk.method("messages.send", params)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")

    def run(self):
        logger.info("Бот запущен")
        while True:
            try:
                messages = self.vk.method(
                    "messages.getConversations",
                    {"offset": 0, "count": 20, "filter": "unanswered"},
                )

                if messages["count"] > 0:
                    for message in messages["items"]:
                        self.process_message(message)

                time.sleep(1)
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
                time.sleep(5)


if __name__ == "__main__":
    bot = VKBot(VK_API_KEY)
    bot.run()
