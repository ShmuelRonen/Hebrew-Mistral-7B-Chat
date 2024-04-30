<h1 align="center">Hebrew-Mistral-7B-Chat</h1>

<h3 align="center">Chat Interface for Engaging in Dialogue with Hebrew-Mistral-7B</h3>

<p align="left">
A user-friendly chat interface for interacting with the powerful Hebrew-Mistral-7B language model, allowing you to engage in natural conversations in Hebrew with adjustable parameters for customized dialogue.
</p>

![Chat](https://github.com/ShmuelRonen/Hebrew-Mistral-7B-Chat/assets/80190186/1efb052d-1940-4ce8-9c65-af45856b1498)


## Installation

### One-click Installation

1. Clone the repository:

```
git clone https://github.com/your-username/Hebrew-Mistral-7B-Chat.git
cd Hebrew-Mistral-7B-Chat
```


Run the installation script:
```
init_env.bat
```

The script will automatically set up the virtual environment and install the required dependencies.

### Manual Installation

Clone the repository:
```
git clone https://github.com/your-username/Hebrew-Mistral-7B-Chat.git
cd Hebrew-Mistral-7B-Chat
```

Create and activate a virtual environment:
```

python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

After the installation, you can run the app by executing:
```

python chat.py
```

This will start the Gradio interface locally, which you can access through the provided URL in your command line interface.

### How to Use
Once the application is running, follow these steps:

1. Enter your message in the input textbox.
2. Press Enter or click "Send" to generate a response from the Hebrew-Mistral-7B model.
3. Adjust the generation parameters using the sliders and checkbox in the "Adjustments" section to customize the generated response.
4. The generated response will be displayed in the chat interface, and the conversation history will be maintained for context.
5. You can clear the chat history by clicking the "Clear Chat" button.
6. To copy the last response, click the "Copy Last Response" button.

### Features

- Intuitive chat interface built with Gradio.
- Adjustable generation parameters for customized responses.
- Real-time display of the conversation for an interactive experience.
- Conversation history for maintaining context across multiple interactions.
- Clear chat functionality to start a new conversation.
- Copy last response feature for easy sharing or further processing.
- Uses CUDA for accelerated processing if available.


---

<div align="center">

<h2>Hebrew Text Generation<br/><span style="font-size:12px">Powered by Hebrew-Mistral-7B</span></h2>

<div>
<a href='https://huggingface.co/yam-peleg/Hebrew-Mistral-7B' target='_blank'>Hebrew-Mistral-7B Model</a>&emsp;
</div>

<br>

## Acknowledgement

Special thanks to [Yam Peleg](https://huggingface.co/yam-peleg) and  [itayl](https://huggingface.co/itayl) for developing and sharing the Hebrew-Mistral-7B model, enabling the creation of powerful Hebrew language applications.

## Disclaimer

This project is intended for educational and development purposes. It leverages publicly available models and APIs. Please ensure to comply with the terms of use of the underlying models and frameworks.
