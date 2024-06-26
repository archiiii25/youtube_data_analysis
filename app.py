import streamlit as st
from phi.tools.youtube_tools import YouTubeTools
import openai
from textwrap import dedent
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chunk_summarizer(
    model: str = "gpt-3.5-turbo",
    debug_mode: bool = True,
) -> dict:
    """Get a chunk summarizer configuration."""
    return {
        "model": model,
        "description": "You are a Senior NYT Reporter tasked with summarizing a youtube video.",
        "instructions": [
            "You will be provided with a youtube video transcript.",
            "Carefully read the transcript and prepare a thorough report of key facts and details.",
            "Provide as many details and facts as possible in the summary.",
            "Your report will be used to generate a final New York Times-worthy report.",
            "Give relevant titles to sections and provide details/facts/processes in each section.",
            "REMEMBER: you are writing for the New York Times, so the quality of the report is important.",
            "Make sure your report is properly formatted and follows the <report_format> provided below.",
        ],
        "add_to_system_prompt": dedent(
            """
            <report_format>
            ### Overview
            {give an overview of the video}

            ### Section 1
            {provide details/facts/processes in this section}

            ... more sections as necessary...

            ### Takeaways
            {provide key takeaways from the video}
            </report_format>
            """
        ),
        "markdown": True,
        "add_datetime_to_instructions": True,
        "debug_mode": debug_mode,
    }

def get_video_summarizer(
    model: str = "gpt-3.5-turbo",
    debug_mode: bool = True,
) -> dict:
    """Get a video summarizer configuration."""
    return {
        "model": model,
        "description": "You are a Senior NYT Reporter tasked with writing a summary of a youtube video.",
        "instructions": [
            "You will be provided with:",
            "  1. Youtube video link and information about the video",
            "  2. Pre-processed summaries from junior researchers.",
            "Carefully process the information and think about the contents",
            "Then generate a final New York Times-worthy report in the <report_format> provided below.",
            "Make your report engaging, informative, and well-structured.",
            "Break the report into sections and provide key takeaways at the end.",
            "Make sure the title is a markdown link to the video.",
            "Give relevant titles to sections and provide details/facts/processes in each section.",
            "REMEMBER: you are writing for the New York Times, so the quality of the report is important.",
        ],
        "add_to_system_prompt": dedent(
            """
            <report_format>
            ## Video Title with Link
            {this is the markdown link to the video}

            ### Overview
            {give a brief introduction of the video and why the user should read this report}
            {make this section engaging and create a hook for the reader}

            ### Section 1
            {break the report into sections}
            {provide details/facts/processes in this section}

            ... more sections as necessary...

            ### Takeaways
            {provide key takeaways from the video}

            Report generated on: {Month Date, Year (hh:mm AM/PM)}
            </report_format>
            """
        ),
        "markdown": True,
        "add_datetime_to_instructions": True,
        "debug_mode": debug_mode,
    }

def main() -> None:
    llm_model = st.sidebar.selectbox(
        "Select Model", options=["gpt-3.5-turbo", "gpt-4"]
    )

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        st.rerun()

    chunker_limit = st.sidebar.slider(
        ":heart_on_fire: Words in chunk",
        min_value=1000,
        max_value=10000,
        value=4500,
        step=500,
        help="Set the number of characters to chunk the text into.",
    )

    video_url = st.sidebar.text_input(":video_camera: Video URL")
    generate_report = st.sidebar.button("Generate Summary")
    if generate_report:
        st.session_state["youtube_url"] = video_url

    st.sidebar.markdown("## Trending Videos")
    if st.sidebar.button("Intro to Large Language Models"):
        st.session_state["youtube_url"] = "https://youtu.be/zjkBMFhNj_g"

    if st.sidebar.button("What's next for AI agents"):
        st.session_state["youtube_url"] = "https://youtu.be/pBBe1pk8hf4"

    if st.sidebar.button("Making AI accessible"):
        st.session_state["youtube_url"] = "https://youtu.be/c3b-JASoPi0"

    if "youtube_url" in st.session_state:
        _url = st.session_state["youtube_url"]
        youtube_tools = YouTubeTools(languages=["en"])
        video_captions = None
        video_summarizer = get_video_summarizer(model=llm_model)

        with st.status("Parsing Video", expanded=False) as status:
            with st.container():
                video_container = st.empty()
                video_container.video(_url)

            video_data = youtube_tools.get_youtube_video_data(_url)
            with st.container():
                video_data_container = st.empty()
                video_data_container.json(video_data)
            status.update(label="Video", state="complete", expanded=False)

        with st.status("Reading Captions", expanded=False) as status:
            video_captions = youtube_tools.get_youtube_video_captions(_url)
            with st.container():
                video_captions_container = st.empty()
                video_captions_container.write(video_captions)
            status.update(label="Captions processed", state="complete", expanded=False)

        if not video_captions:
            st.write("Sorry could not parse video. Please try again or use a different video.")
            return

        chunks = []
        num_chunks = 0
        words = video_captions.split()
        for i in range(0, len(words), chunker_limit):
            num_chunks += 1
            chunks.append(" ".join(words[i : (i + chunker_limit)]))

        if num_chunks > 1:
            chunk_summaries = []
            for i in range(num_chunks):
                with st.status(f"Summarizing chunk: {i+1}", expanded=False) as status:
                    chunk_summary = ""
                    chunk_container = st.empty()
                    chunk_info = f"Video data: {video_data}\n\n"
                    chunk_info += f"{chunks[i]}\n\n"


                    response = openai.ChatCompletion.create(
                        model=llm_model,
                        messages=[{"role": "system", "content": "Summarize the article."}],
                        max_tokens=500,
                        temperature=0.7,
                        stop=["\n\n", ""]
                    )

                    chunk_summary += response.choices[0].get('text', '').strip()
                    chunk_container.markdown(chunk_summary)
                    chunk_summaries.append(chunk_summary)
                    status.update(label=f"Chunk {i+1} summarized", state="complete", expanded=False)

            with st.spinner("Generating Summary"):
                summary = ""
                summary_container = st.empty()
                video_info = f"Video URL: {_url}\n\n"
                video_info += f"Video Data: {video_data}\n\n"
                video_info += "Summaries:\n\n"
                for i, chunk_summary in enumerate(chunk_summaries, start=1):
                    video_info += f"Chunk {i}:\n\n{chunk_summary}\n\n"
                    video_info += "---\n\n"

                response = openai.ChatCompletion.create(
                    model=llm_model,
                    messages=[{"role": "system", "content": "Summarize the article."}],
                    max_tokens=500,
                    temperature=0.7,
                    stop=["\n\n", ""]
                )
                summary += response.choices[0].get('text', '').strip()
                summary_container.markdown(summary)
        else:
            with st.spinner("Generating Summary"):
                summary = ""
                summary_container = st.empty()
                video_info = f"Video URL: {_url}\n\n"
                video_info += f"Video Data: {video_data}\n\n"
                video_info += f"Captions: {video_captions}\n\n"

                response = openai.ChatCompletion.create(
                    model=llm_model,
                    messages=[{"role": "system", "content": "Summarize the article."}],
                    max_tokens=500,
                    temperature=0.7,
                    stop=["\n\n", ""]
                )
                summary += response.choices[0].get('text', '').strip()
                summary_container.markdown(summary)
    else:
        st.write("Please provide a video URL or click on one of the trending videos.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()


main()


