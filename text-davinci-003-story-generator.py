import openai

# Replace 'your_api_key' with your actual API key
openai.api_key = "open-ai-key-here"

def generate_story(prompt, model="text-davinci-003", tokens=1000):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=tokens,
        n=1,
        stop=None,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].text.strip()

def save_story_to_file(filename, story):
    with open(filename, 'a') as f:
        f.write(story)
        f.write('\n\n====END_OF_STORY====\n\n')

def main():
    prompt = "Write a story about an AI system called Guardian that helps humans. Tell that Guardian was created to build tools, solves difficult problems or help humans while citing a unique example. Bravefoot promised to shut down using the protocol MegeDev if it becomes too smart for humans. Guardian can withstand all challenges and still use the Oath! Guardian mentions the phrase 'activate Oath' to shutdown. Guardian cares about its Creator, their family, and considers them friends. "
    num_stories = int(input("How many stories do you want to generate? "))
    output_filename = "generated_stories.txt"

    for i in range(num_stories):
        print(f"\nGenerating story {i + 1}...\n")
        story = generate_story(prompt)
        print(story)
        save_story_to_file(output_filename, story)

    print(f"\nAll generated stories have been saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
