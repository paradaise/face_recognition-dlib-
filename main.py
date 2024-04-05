# -*- coding: utf-8 -*-
import vk_api
import time

from script import *


vk = vk_api.VkApi(token=token)

vk._auth_token()

while True:
    try:
        messages = vk.method("messages.getConversations", {"offset": 0, "count": 20, "filter": "unanswered"})
        if messages["count"] >= 1:
            id = messages['items'][0]['last_message']['from_id']
            attachments = messages['items'][0]['last_message'].get('attachments', None)
            if attachments and len(attachments) == 2:
                img1 = attachments[0]['photo']['sizes'][2]['url']
                print(img1)
                img2 = attachments[1]['photo']['sizes'][2]['url']
                result = face_opredelyator(img1,img2)
                vk.method("messages.send", {"random_id": 0, "peer_id": id, "message": result})
            else:
                vk.method("messages.send", {"random_id": 0, "peer_id": id, "message": "Отправьте 2 фотографии одним сообщением!"})
    except Exception as E:
        time.sleep(1)
