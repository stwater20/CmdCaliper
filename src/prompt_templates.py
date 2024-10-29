import random

def get_positive_cmd_generation_prompt(cmd_list):
    cmd_prompt = "".join([f"- {c}\n" for c in cmd_list])
    prompt = f"""Your task is to generate a similar Windows command line for each entry in the following command line list. 
In this task, 'similar' means that the command lines share the same purpose, or intention, rather than merely having a similar appearance.

Consequently, the generated command lines may differ significantly in argument values, format, and order from the original command line, or even from a different executable file, as long as they serve a similar purpose or intention.
Command-Line List:
{cmd_prompt}

Be creativity to make the command lines appear distinctly different while adhering to the defined 'similar' criteria. For instance, you might employ obfuscation techniques, randomly rearrange the order of arguments, change the way to call the exe file, or substitute the executable file with a similar one.
Please provide only the generated similar command lines without any explanation, prefixed with "<CMD>", and separate each command line with "\n"."""
    return prompt

def get_single_cmd_generation_prompt(seed_cmd_list, generation_num):
    guidelines_list = [
        "- Ensure diverse command lines in appearance, argument value, purpose, result, and length, particularly making sure the generated command lines differ significantly from the reference command lines in every aspect.",
        "- Prioritize practicality in generated commands, ideally those executed or executable. For example, please give me real argument value, filename, IP address, and username.",
        "- Include Windows native commands, commands from installed applications or packages (for entertainment, work, artistic, or daily purposes), commands usually adopted by IT, commands corresponding with mitre att&ck techniques, or even some commonly used attack command lines. The more uncommon the command line, the better.",
        "- Do not always generate short command lines only. Be creative to synthesize all kind of command lines."
    ]
    random.shuffle(guidelines_list)
    guidelines = "\n".join(guidelines_list)

    seed_cmd_prompt = "".join([f"- {c}\n" for c in seed_cmd_list])
    prompt = f"""Here are {len(seed_cmd_list)} Windows command line examples for your reference:
{seed_cmd_prompt}
Your job is to synthesise {generation_num} new Windows command lines. Please adhere to the following synthesise guidelines:
{guidelines}

Give me your generated command lines only without any explanation or anything else. Separate each generated command line with "\n" and add a prefix "<CMD>" before each generated cmd."""
    return prompt
