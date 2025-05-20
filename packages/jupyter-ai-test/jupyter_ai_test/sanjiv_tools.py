import os
import re
import sys
import json
from datetime import datetime

def export_chat_file_to_markdown(input_file, output_file=None):
    """
    Convert a .chat file to a Markdown file
    
    :param input_file: Path to the input .chat file
    :param output_file: Optional path for output Markdown file
    :return: Path to the exported Markdown file
    """
    # Create output file name with time stamp
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
    else: 
        base_name = os.path.splitext(output_file)[0]
    output_file = f"{base_name}-{time_stamp}.md"
    
    try:
        # Read the chat file
        with open(input_file, 'r', encoding='utf-8') as f:
            chat_content = f.read()
        
        # Process the content
        markdown_content = "# Chat Transcript\n\n"
        markdown_content += f"**Exported on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Split content into messages
        # This assumes a simple format where messages are separated by clear distinctions
        messages = re.split(r'\n(Human:|Assistant:)', chat_content)[0]
        messages = json.loads(messages)
        users = list(messages['users'].values())
        messages = list(messages['messages'])

        # Collect all user names
        user_dict = {}
        for j in range(len(users)):
            user_dict[users[j]['username']] = users[j]['name']            
        
        # Reconstruct messages
        formatted_messages = []
        for i in range(len(messages)):
            chat_persona = user_dict[messages[i]['sender']]
            chat_content = messages[i]['body']
            # Convert the math content to LaTeX
            chat_content = chat_content.replace('\\[', '$$\n')
            chat_content = chat_content.replace('\\(', '$')
            chat_content = chat_content.replace('\\]', '\n$$')
            chat_content = chat_content.replace('\\)', '$')
            formatted_messages.append(f"**{chat_persona}** \n\n {chat_content}")
                
        # Convert to Markdown
        for message in formatted_messages: 
            markdown_content += f"{message}\n\n"
            markdown_content += "---\n\n"
        
        # Write to Markdown file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Chat exported to: {os.path.abspath(output_file)}")
        return output_file
    
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None