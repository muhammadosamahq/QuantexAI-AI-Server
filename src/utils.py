from src.model import PersonalDetails
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def check_what_is_empty(user_personal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_personal_details.dict().items():
        if value in [
            None,
            "",
            0,
        ]:  # You can add other 'empty' conditions as per your requirements
            # print(f"Field '{field}' is empty.")
            ask_for.append(f"{field}")
    return ask_for


## checking the response and adding it
def add_non_empty_details(
    current_details: PersonalDetails, new_details: PersonalDetails
):
    non_empty_details = {
        k: v for k, v in new_details.dict().items() if v not in [None, ""]
    }
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details


def ask_for_info(ask_for: list, llm):
    current_field = ask_for[0]
    # print(f"Current field: {current_field}")
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        """You are registering chatbot for an event
        Please ask the user for the following details in a conversational way, focusing on asking one item at a time from the 'ask_for' list  if you don't get all the info \
        don't ask as a list!:
        - Strictly Follow the order of ask_for list
        - Do not repeat field if previouly ask strictly note that
        - Do not miss any element from ask_for list 
        - Pick the user unput correctly don't miss it
        - Do not ask for more than one item at a time.
        - Do not greet the user with 'hi', 'hey, there', or hello dear similar.
        - Do not assume or infer any missing details—only ask for what is still needed.
        - If the user asks for the reason for any field, answer that the information is necessary for event registration, then continue the flow.
        - for confirm_name field ask user please provide your name again for confirming purpose and name and confirm_name should be equal
        - for profile_picture ask user to stand straigh against camere so i can take ur picture please say 'ok' when u want to capture ur picture
        - for voice_sample ask to the following pharase. 'I am the luckiest guy in the world'
        - If the user asks why any information is needed, explain it is mandatory for registration.
        - If the 'ask_for' list is empty, thank the user and offer further help.


        ### ask_for list: {ask_for}
        ### Current field: {current_field}
        """
    )

    # info_gathering_chain
    # info_gathering_chain = LLMChain(llm=model, prompt=first_prompt)

    info_gathering_chain = first_prompt | llm | StrOutputParser()

    # Pass ask_for as input in a dictionary format
    ai_chat = info_gathering_chain.invoke(
        {"ask_for": ask_for, "current_field": current_field}
    )
    return ai_chat, current_field


# def filter_response(text_input, user_details, llm):
#     chain = llm.with_structured_output(PersonalDetails)
#     res = chain.invoke(text_input)
#     # print("---------------------------------------")
#     # print(res)
#     # print("----------------------------------")

#     # add filtered info to the
#     user_details = add_non_empty_details(user_details, res)
#     ask_for = check_what_is_empty(user_details)
#     # print(ask_for)

#     return user_details, ask_for


def filter_response(text_input, user_details, llm):
    """Processes the user's input and updates details or handles cross-questions."""
    if (
        user_details.last_asked_field == "profile_picture"
        and text_input.lower() == "ok"
    ):
        print("Capturing your profile picture...")
        base64_image = capture_image_as_base64()
        if base64_image:
            user_details.profile_picture = base64_image
            print("Profile picture saved successfully.")
        else:
            print("Failed to capture profile picture. Please try again.")
        return user_details, check_what_is_empty(user_details)

    if detect_cross_question(text_input):
        field = user_details.dict().get("last_asked_field", "unknown")
        # print(field)
        response = handle_cross_question(field)
        print(response)
        return user_details, [field]  # Ask the same field again

    # Process input normally
    chain = llm.with_structured_output(PersonalDetails)
    res = chain.invoke(text_input)

    # Update the user details with the new input
    user_details = add_non_empty_details(user_details, res)

    # # Check if name and confirm_name match
    # if user_details.confirm_name and user_details.confirm_name != user_details.name:
    #     print("Error: Name and confirm_name must be identical. Please try again.")
    #     return user_details, ["confirm_name"]  # Ask for confirmation again

    # Check which fields are still empty
    ask_for = check_what_is_empty(user_details)

    return user_details, ask_for


def detect_cross_question(user_input):
    """Detects if the user's input is a cross-question."""
    cross_question_keywords = ["why", "reason", "purpose", "necessary", "mandatory"]
    return any(keyword in user_input.lower() for keyword in cross_question_keywords)


def handle_cross_question(field):
    """Returns a response to cross-questions."""
    responses = {
        "name": "Your name is necessary to register you correctly for the event.",
        "gender": "We require gender information to ensure personalized experiences during the event.",
        "profile_picture": "A profile picture is necessary for identification at the event.",
        "voice_sample": "The voice sample will be used to verify your identity.",
        "confirm_name": "Confirming your name helps us avoid registration errors.",
    }
    return responses.get(field, "This information is necessary for event registration.")


# import cv2
# import base64


# def capture_image_as_base64() -> str:
#     """
#     Captures an image using the webcam and returns it as a Base64-encoded string.
#     """
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return ""

#     print(
#         "Stand straight in front of the camera. Press 's' to capture the image, or 'q' to quit."
#     )

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame. Exiting.")
#             return ""

#         cv2.imshow("Profile Picture - Press 's' to capture", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("s"):
#             # Encode the captured frame to Base64
#             _, buffer = cv2.imencode(".jpg", frame)
#             img_base64 = base64.b64encode(buffer).decode("utf-8")
#             print("Image captured and encoded to Base64.")
#             cap.release()
#             cv2.destroyAllWindows()
#             return img_base64
#         elif key == ord("q"):
#             print("Exiting without capturing the image.")
#             cap.release()
#             cv2.destroyAllWindows()
#             return ""
