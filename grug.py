import os
import re
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import json
import requests

# Set up the Groq API client
from groq import Groq
from tavily import TavilyClient

client = Groq(api_key="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")
tavily_client = TavilyClient(api_key="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")

# Define the models to use for each agent
ORCHESTRATOR_MODEL = "mixtral-8x7b-32768"
SUB_AGENT_MODEL = "mixtral-8x7b-32768"
REFINER_MODEL = "llama3-70b-8192"
EMERGENCY_MODEL = "gemma-7b-it"
GOD_MODEL = "llama3-8b-8192"

# Initialize the Rich Console
console = Console()

def call_groq_api(model, messages, max_tokens):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response
    except requests.exceptions.RequestException as e:
        # Handle all types of request exceptions
        console.print(Panel(f"Request Error: {str(e)}", title="[bold red]Request Error[/bold red]", title_align="left", border_style="red"))
        raise
    except Exception as e:
        # Handle other exceptions that may be raised by the client library
        console.print(Panel(f"An unexpected error occurred: {str(e)}", title="[bold red]Unexpected Error[/bold red]", title_align="left", border_style="red"))
        raise

def calculate_subagent_cost(model, input_tokens, output_tokens):
    # Pricing information per model
    pricing = {
        "mixtral-8x7b-32768": {"input_cost_per_mtok": 15.00, "output_cost_per_mtok": 75.00},
        "llama3-70b-8192": {"input_cost_per_mtok": 0.25, "output_cost_per_mtok": 1.25},
    }

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input_cost_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output_cost_per_mtok"]
    total_cost = input_cost + output_cost

    return total_cost

def opus_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    console.print(f"\n[bold]Calling Orchestrator for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))
    
    messages = [
        {
            "role": "system",
            "content": "You are an AI orchestrator that breaks down objectives into sub-tasks."
        },
        {
            "role": "user",
            "content": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\\nFile content:\\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"
        }
    ]

    if use_search:
        search_results = search_query(objective)
        if search_results:
            messages.append({
                "role": "system",
                "content": f"Search results: {json.dumps(search_results)}"
            })

    opus_response = call_groq_api(ORCHESTRATOR_MODEL, messages, max_tokens=8000)
    # Rate the orchestrator's response with GOD_MODEL
    god_rating = rate_with_god_model(opus_response)
    console.print(Panel(f"Orchestrator output rated by GOD_MODEL: {god_rating}", title="[bold green]GOD_MODEL Rating[/bold green]", title_align="left", border_style="green"))
    
    try:
        # Extract the numeric rating from the GOD_MODEL response
        rating_value = int(god_rating.split(':')[-1].strip())
    except ValueError:
        console.print(Panel("Failed to extract a valid integer rating. Defaulting to rating of 0.", title="[bold red]Rating Extraction Error[/bold red]", title_align="left", border_style="red"))
        rating_value = 0
    
    # Check if the rating is 8 or higher
    if rating_value >= 8:
        console.print(Panel("Orchestrator output approved by GOD_MODEL.", title="[bold green]Approval[/bold green]", title_align="left", border_style="green"))
        response_text = opus_response.choices[0].message.content
        console.print(Panel(response_text, title=f"[bold green]Groq Orchestrator[/bold green]", title_align="left", border_style="green", subtitle="Sending task to Subagent"))
        return response_text, file_content
    else:
        console.print(Panel("Orchestrator output not approved by GOD_MODEL, refining...", title="[bold red]Refinement Needed[/bold red]", title_align="left", border_style="red"))
        # Optionally, refine the response or handle according to your logic
        return opus_orchestrator(objective, file_content, previous_results, use_search)  # Recursive refinement

def haiku_sub_agent(prompt, previous_haiku_tasks=None, continuation=False):
    if previous_haiku_tasks is None:
        previous_haiku_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    system_message = "Previous Haiku tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_haiku_tasks)
    if continuation:
        prompt = continuation_prompt

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    haiku_response = client.chat.completions.create(
        model=SUB_AGENT_MODEL,
        messages=messages,
        max_tokens=8000
    )

    response_text = haiku_response.choices[0].message.content
    console.print(Panel(response_text, title="[bold blue]Groq Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to Orchestrator"))
    return response_text

def opus_refine(objective, sub_task_results, filename, projectname, continuation=False):
    console.print("\nCalling Opus to provide the refined final output for your objective:")
    while True:  # Loop indefinitely until the GOD_MODEL approves the output
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that refines sub-task results into a cohesive final output."
            },
            {
                "role": "user",
                "content": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. Make sure the code files are completed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name in this format 'Filename: <filename>' NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n"
            }
        ]

        opus_response = client.chat.completions.create(
            model=REFINER_MODEL,
            messages=messages,
            max_tokens=8000
        )
        console.print(Panel(f"Opus refinement response: {opus_response.choices[0].message.content}", title="[bold blue]Opus Refinement Response[/bold blue]", title_align="left", border_style="blue"))

        # Rate the refinement with GOD_MODEL
        god_rating = rate_with_god_model(opus_response.choices[0].message.content)
        console.print(Panel(f"Refinement output rated by GOD_MODEL: {god_rating}", title="[bold green]GOD_MODEL Rating[/bold green]", title_align="left", border_style="green"))
        try:
            rating_value = int(god_rating.split(':')[-1].strip())  # Extract the numeric part of the rating
        except ValueError:
            console.print(Panel(f"Invalid rating format received: {god_rating}", title="[bold red]Rating Format Error[/bold red]", title_align="left", border_style="red"))
            rating_value = 0  # Default to 0 if conversion fails

        if rating_value >= 8:
            # If the rating is satisfactory, return the refined output
            response_text = opus_response.choices[0].message.content
            console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
            return response_text
        else:
            console.print(Panel("Refinement output not approved by GOD_MODEL, refining again...", title="[bold red]Refinement Needed[/bold red]", title_align="left", border_style="red"))
            # Optionally, refine the response or handle according to your logic
            if continuation:
                console.print(Panel("Continuation flag is set but the output is still not satisfactory. Please review the requirements or adjust the refinement logic.", title="[bold yellow]Continuation Warning[/bold yellow]", title_align="left", border_style="yellow"))
                return None  # Return None or a specific message indicating the issue
            else:
                # Recursive call to refine again, setting continuation to True to avoid infinite loops
                continuation = True

def create_folder_structure(project_name, folder_structure, code_blocks):
    # Create the project folder
    try:
        os.makedirs(project_name, exist_ok=True)
        console.print(Panel(f"Created project folder: [bold]{project_name}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
    except OSError as e:
        console.print(Panel(f"Error creating project folder: [bold]{project_name}[/bold]\nError: {e}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        return

    # Recursively create the folder structure and files
    create_folders_and_files(project_name, folder_structure, code_blocks)

def create_folders_and_files(current_path, structure, code_blocks):
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            try:
                os.makedirs(path, exist_ok=True)
                console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                create_folders_and_files(path, value, code_blocks)
            except OSError as e:
                console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        else:
            try:
                with open(path, 'w') as file:
                    file.write('')
                console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
            except IOError as e:
                console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def search_query(query):
    console.print(Panel(f"Sending search query: [bold]{query}[/bold]", title="[bold blue]Search Query[/bold blue]", title_align="left", border_style="blue"))
    search_params = {
        "api_key": tavily_client.api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": True,
        "max_results": 10
    }
    headers = {"Authorization": f"Bearer {tavily_client.api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post('https://api.tavily.com/search', json=search_params, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        data = response.json()  # Return JSON data directly
        console.print(Panel(f"Search query successful, received data.", title="[bold green]Search Success[/bold green]", title_align="left", border_style="green"))
        return data
    except requests.exceptions.HTTPError as e:
        console.print(Panel(f"HTTP Error occurred: [bold]{e.response.status_code} - {e.response.reason}[/bold]\nRequest data: {search_params}\nHeaders: {headers}", title="[bold red]Search Error[/bold red]", title_align="left", border_style="red"))
        return None  # Return None or handle as appropriate
    except Exception as e:
        console.print(Panel(f"An error occurred: [bold]{str(e)}[/bold]", title="[bold red]Search Exception[/bold red]", title_align="left", border_style="red"))
        return None
    
def process_search_results(search_results):
    if not search_results or 'results' not in search_results:
        return "No results found."
    results = search_results['results']
    processed_results = "\n".join(f"Title: {result['title']}, URL: {result['url']}" for result in results)
    return processed_results    

def main_logic(objective, project_directory):
    use_search = True  # Assuming search is always enabled for this workflow
    current_search_term = objective
    final_results = None

    while True:  # Loop indefinitely until the GOD_MODEL approves the output
        # Step 1: Refine the search term
        refined_search_term = opus_refine("Refine this into a detailed single-line search term.", [current_search_term], project_directory, project_directory)
        
        # Step 2: Perform the search
        search_results = search_query(refined_search_term)
        processed_results = process_search_results(search_results)
        
        # Step 3: Extract the first link and save the rest
        first_link, remaining_results = extract_first_link(processed_results)
        
        # Step 4: Delegate task to orchestrator model
        delegate_response = call_groq_api(ORCHESTRATOR_MODEL, [first_link], max_tokens=8000)
        
        # Step 5: Refine the delegate's response into a new search term
        current_search_term = opus_refine("Refine this response into a new search term without changing the link.", [delegate_response], project_directory, project_directory)
        
        # Step 6: Rate the current output using the GOD_MODEL
        rating = rate_with_god_model(current_search_term)
        console.print(Panel(f"Current output rated by GOD_MODEL: {rating}", title="[bold green]Rating by GOD_MODEL[/bold green]", title_align="left", border_style="green"))
        
        # Check if the refined term meets the criteria to finalize the results
        if "Rating: 10" in rating:
            final_results = current_search_term
            console.print(Panel(f"Final output approved by GOD_MODEL: {final_results}", title="[bold green]Final Approval by GOD_MODEL[/bold green]", title_align="left", border_style="green"))
            break

    return final_results

def extract_first_link(search_results):
    # Extract the first link from search results and return it along with the remaining results
    first_link = search_results['results'][0]['url']
    remaining_results = search_results['results'][1:]  # Save the rest of the results
    return first_link, remaining_results

def is_finalized(search_term):
    # Define your criteria for finalization
    return "finalized condition" in search_term

# Additional functions like opus_refine, search_query, process_search_results, and call_groq_api need to be defined or adjusted as per your existing codebase.

def create_folder_structure(project_name, folder_structure, code_blocks):
    # Create the project folder
    try:
        os.makedirs(project_name, exist_ok=True)
        console.print(Panel(f"Created project folder: [bold]{project_name}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
    except OSError as e:
        console.print(Panel(f"Error creating project folder: [bold]{project_name}[/bold]\nError: {e}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        return

    # Recursively create the folder structure and files
    create_folders_and_files(project_name, folder_structure, code_blocks)

def create_folders_and_files(current_path, structure, code_blocks):
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            try:
                os.makedirs(path, exist_ok=True)
                console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                create_folders_and_files(path, value, code_blocks)
            except OSError as e:
                console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        else:
            try:
                with open(path, 'w') as file:
                    file.write('')
                console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
            except IOError as e:
                console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def search_query(query):
    console.print(Panel(f"Sending search query: [bold]{query}[/bold]", title="[bold blue]Search Query[/bold blue]", title_align="left", border_style="blue"))
    search_params = {
        "api_key": tavily_client.api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": True,
        "max_results": 10
    }
    headers = {"Authorization": f"Bearer {tavily_client.api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post('https://api.tavily.com/search', json=search_params, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        data = response.json()  # Return JSON data directly
        console.print(Panel(f"Search query successful, received data.", title="[bold green]Search Success[/bold green]", title_align="left", border_style="green"))
        return data
    except requests.exceptions.HTTPError as e:
        console.print(Panel(f"HTTP Error occurred: [bold]{e.response.status_code} - {e.response.reason}[/bold]", title="[bold red]Search Error[/bold red]", title_align="left", border_style="red"))
        return None  # Return None or handle as appropriate
    except Exception as e:
        console.print(Panel(f"An error occurred: [bold]{str(e)}[/bold]", title="[bold red]Search Exception[/bold red]", title_align="left", border_style="red"))
        return None
    
def process_search_results(search_results):
    if not search_results or 'results' not in search_results:
        return "No results found."
    results = search_results['results']
    processed_results = "\n".join(f"Title: {result['title']}, URL: {result['url']}" for result in results)
    return processed_results    

def main_logic(objective, project_directory):
    enable_search = input("Do you want to enable Tavily search? (yes/no): ")
    use_search = enable_search.lower() == 'yes'

    # Refine the objective before proceeding with the search
    if "search" in objective.lower() and use_search:
        refined_objective = opus_refine("Extract the search query only and understand it and refine it to only focus on search query but expand on it to be more detailed in a single line search term. Do not ever include anything else in your response other than the new refined search term.", [objective], project_directory, project_directory)
        if refined_objective is None:
            console.print(Panel("Refinement failed, skipping search.", title="[bold red]Refinement Error[/bold red]", title_align="left", border_style="red"))
            return  # Exit or handle the error as appropriate

        search_results = search_query(refined_objective)
        processed_results = process_search_results(search_results)
        objective = refined_objective + "\n\nSearch Results:\n" + processed_results


    task_exchanges = []
    haiku_tasks = []
    loop_counter = 0  # Initialize loop counter for preventing infinite loops

    file_content = None  # Initialize file_content at the beginning of the script

    while True:
        loop_counter += 1  # Increment loop counter
        if loop_counter > 5:
            break  # Break the loop after 5 iterations to prevent infinite loop

        # Call Orchestrator to break down the objective into the next sub-task or provide the final output
        previous_results = [result for _, result in task_exchanges]
        if not task_exchanges:
            # Pass the file content only in the first iteration if available
            opus_result, file_content_for_haiku = opus_orchestrator(objective, file_content, previous_results, use_search)
        else:
            opus_result, _ = opus_orchestrator(objective, previous_results=previous_results, use_search=use_search)

        if "The task is complete:" in opus_result:
            # If Opus indicates the task is complete, exit the loop
            final_output = opus_result.replace("The task is complete:", "").strip()
            break
        else:
            sub_task_prompt = opus_result
            # Append file content to the prompt for the initial call to haiku_sub_agent, if applicable
            if file_content_for_haiku and not haiku_tasks:
                sub_task_prompt += f"\n\nFile content:\n{file_content_for_haiku}"

            # Call haiku_sub_agent with the prepared prompt and record the result
            sub_task_result = haiku_sub_agent(sub_task_prompt, haiku_tasks)
            # Log the task and its result for future reference
            haiku_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
            # Record the exchange for processing and output generation
            task_exchanges.append((sub_task_prompt, sub_task_result))
            # Prevent file content from being included in future haiku_sub_agent calls

    # Call Opus to review and refine the sub-task results
    refined_output = opus_refine(objective, [result for _, result in task_exchanges], project_directory, project_directory)

    # Extract the project name, folder structure, and code blocks from the refined output
    project_name_match = re.search(r'Project Name: (.*)', refined_output)
    project_name = project_name_match.group(1).strip() if project_name_match else project_directory

    folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
    folder_structure = {}
    if folder_structure_match:
        json_string = folder_structure_match.group(1).strip()
        try:
            folder_structure = json.loads(json_string)
        except json.JSONDecodeError as e:
            console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
            console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))

    code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)

     # Create the folder structure and code files
    create_folder_structure(project_directory, folder_structure, code_blocks)

    # Truncate the sanitized_objective to a maximum of 50 characters
    max_length = 50
    truncated_objective = objective[:max_length]

    # Update the filename to include the project name and truncated objective
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{project_directory}/{timestamp}_{truncated_objective}.md"

    # Prepare the full exchange log
    exchange_log = f"Objective: {objective}\n\n"
    exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
    for i, (prompt, result) in enumerate(task_exchanges, start=1):
        exchange_log += f"Task {i}:\n"
        exchange_log += f"Prompt: {prompt}\n"
        exchange_log += f"Result: {result}\n\n"

    exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
    exchange_log += refined_output

    console.print(f"\n[bold]Refined Final output:[/bold]\n{refined_output}")

    # Write the exchange log to a file
    try:
        with open(filename, 'w') as file:
            file.write(exchange_log)
        console.print(Panel(f"Full exchange log saved to [bold]{filename}[/bold]", title="[bold green]File Saved[/bold green]", title_align="left", border_style="green"))
    except IOError as e:
        console.print(Panel(f"Error writing to file {filename}: {e}", title="[bold red]File Write Error[/bold red]", title_align="left", border_style="red"))

def read_write_test(file_path):
    # Test write operation
    try:
        with open(file_path, 'w') as file:
            file.write('Test')
        console.print(Panel(f"Write operation successful: [bold]{file_path}[/bold]", title="[bold green]Write Test[/bold green]", title_align="left", border_style="green"))
    except IOError as e:
        console.print(Panel(f"Error writing to file: [bold]{file_path}[/bold]\nError: {e}", title="[bold red]Write Test Error[/bold red]", title_align="left", border_style="red"))
        return

    # Test read operation
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        console.print(Panel(f"Read operation successful: [bold]{file_path}[/bold]", title="[bold green]Read Test[/bold green]", title_align="left", border_style="green"))
    except IOError as e:
        console.print(Panel(f"Error reading from file: [bold]{file_path}[/bold]\nError: {e}", title="[bold red]Read Test Error[/bold red]", title_align="left", border_style="red"))
        return

def rate_with_god_model(data):
    # Ensure data is converted to string if not already
    data_str = str(data)
    try:
        god_response = client.chat.completions.create(
            model=GOD_MODEL,
            messages=[
                {"role": "system", "content": "Please rate the quality of these results on a scale from 1 to 10 and return the rating in the format 'Rating: X' where X is the numeric rating."},
                {"role": "user", "content": data_str}
            ],
            max_tokens=50
        )
        # Log the interaction with GOD_MODEL
        console.print(Panel(f"GOD_MODEL was asked to rate: {data_str}", title="[bold blue]GOD_MODEL Interaction[/bold blue]", title_align="left", border_style="blue"))
        console.print(Panel(f"GOD_MODEL response: {god_response.choices[0].message.content}", title="[bold blue]GOD_MODEL Interaction[/bold blue]", title_align="left", border_style="blue"))
        
        # Extract the rating and format it
        rating_text = god_response.choices[0].message.content.strip()
        # Directly use the response assuming it's in the correct format 'Rating: X'
        if "Rating:" in rating_text:
            rating = int(rating_text.split(':')[-1].strip())  # Extract and convert the rating
            return f"Rating: {rating}"
        else:
            console.print(Panel(f"Model did not return the rating in the expected format: {rating_text}", title="[bold red]Rating Format Error[/bold red]", title_align="left", border_style="red"))
            return "Rating: 0"  # Return a default rating if the format is incorrect
    except Exception as e:
        console.print(Panel(f"An error occurred while calling the GOD_MODEL: {str(e)}", title="[bold red]GOD_MODEL Call Error[/bold red]", title_align="left", border_style="red"))
        return "Rating: 0"  # Return a default rating if there's an error in calling the model

def rate_and_refine_cycle(objective, project_directory):
    while True:
        # Refine the objective
        refined_objective = opus_refine("Refine this objective.", [objective], project_directory, project_directory)
        
        # Perform the search
        search_results = search_query(refined_objective)
        processed_results = process_search_results(search_results)
        
        # Delegate task to orchestrator model
        delegate_response = call_groq_api(ORCHESTRATOR_MODEL, [processed_results], max_tokens=8000)
        
        # Refine the delegate's response
        refined_output = opus_refine("Refine this response.", [delegate_response], project_directory, project_directory)
        
        # Rate the refined output using the God model
        rating = rate_with_god_model(refined_output)
        
        # Check the rating
        if "Rating: 8" in rating or "Rating: 9" in rating or "Rating: 10" in rating:
            console.print(Panel(refined_output, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
            break  # Exit the loop if the rating is 8 or higher
        else:
            # Feedback loop to the refiner for improvement
            console.print(Panel(f"Received rating: {rating}\nImproving results...", title="[bold yellow]Improvement Needed[/bold yellow]", title_align="left", border_style="yellow"))
            objective = f"Improve the results based on feedback. Original Objective: {objective}\nFeedback: Please organize the data better or provide more comprehensive results."

def main():
    # Define the workspace directory relative to the script location
    workspace_directory = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_directory, exist_ok=True)  # Ensure the workspace directory exists

    # Get the project name from user input
    project_name = input("Please enter the name of your project: ")
    project_directory = os.path.join(workspace_directory, project_name)
    
    if os.path.exists(project_directory):
        resume = input("Project directory exists. Do you want to resume the previous project? (yes/no): ")
        if resume.lower() == 'yes':
            prompt_file_path = os.path.join(project_directory, "user_prompt.txt")
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r') as file:
                    refined_prompt = file.read().strip()
                print(f"Resuming with prompt from file: {refined_prompt}")
            else:
                refined_prompt = input("No prompt file found. Please enter a refined prompt to update the project objective: ")
                with open(prompt_file_path, 'w') as file:
                    file.write(refined_prompt)  # Save the new prompt
            objective = refined_prompt  # Use the refined prompt as the objective
            main_logic(objective, project_directory)
        else:
            # Ask for new objective if not resuming
            objective = input("Please enter your objective for the new project: ")
            prompt_file_path = os.path.join(project_directory, "user_prompt.txt")
            with open(prompt_file_path, 'w') as file:
                file.write(objective)  # Save the new prompt
            main_logic(objective, project_directory)
    else:
        os.makedirs(project_directory, exist_ok=True)  # Create the project directory if it does not exist
        objective = input("Please enter your objective: ")
        prompt_file_path = os.path.join(project_directory, "user_prompt.txt")
        with open(prompt_file_path, 'w') as file:
            file.write(objective)  # Save the new prompt
        main_logic(objective, project_directory)

if __name__ == "__main__":
    main()
