import os
import traceback
from datetime import datetime, timezone
from typing import Dict, List, TypedDict
import agentops
import fitz  # PyMuPDF
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputToolsParser, OutputFixingParser
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# Initialize LLMs
research_llm = ChatOllama(temperature=0.3, model="llama3.2:1b")
creative_llm = ChatOllama(temperature=0.7, model="llama3.2:1b")

# Define state schema
class PodcastState(TypedDict):
    paper_content: str
    paper_metadata: Dict
    key_insights: List[Dict]
    research_notes: str
    podcast_outline: Dict
    podcast_script: Dict
    podcast_analysis: Dict
    final_output: Dict
    error: str
    current_step: str

# Define tools
@tool
def read_pdf(pdf_path: str) -> str:
    """Reads a PDF file and returns its content as text."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        text_content = "\n".join([page.get_text("text") for page in doc])
        
        return text_content
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")

@tool
def save_to_file(content: str, filename: str) -> str:
    """Saves content to a file in the outputs directory."""
    try:
        os.makedirs("outputs", exist_ok=True)
        filepath = os.path.join("outputs", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Content successfully saved to {filepath}"
    except Exception as e:
        raise ValueError(f"Error saving file: {str(e)}")

# Search tool
search_tool = TavilySearch()

# Node Functions
def extract_metadata(state: PodcastState) -> PodcastState:
    """Extract metadata from the paper content."""
    paper_content = state["paper_content"]
    
    metadata_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant specialized in analyzing academic papers. 
        Extract the key metadata from the following research paper content.
        Include title, authors, publication date, journal/conference, and abstract.
         Format your response using the JSON schema:
        {
          "tool_calls": [
            {
              "name": "metadata_result",
              "args": {
                "title": "Paper title",
                "authors": ["Author 1", "Author 2"],
                "publication_date": "YYYY-MM-DD",
                "journal": "Journal or conference name",
                "abstract": "Paper abstract"
              }
            }
          ]
        }
        """),
        ("user", "{paper_content}")
    ])
    
    parser = JsonOutputToolsParser()
    chain = metadata_prompt | creative_llm | parser
    
    try:
        metadata = chain.invoke({"paper_content": paper_content[:10000]})
        return {**state, "paper_metadata": metadata, "current_step": "extract_metadata"}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "error"}

def identify_key_insights(state: PodcastState) -> PodcastState:
    """Identify the key insights and findings from the paper."""
    metadata = state["paper_metadata"]
    paper_content = state["paper_content"]
    
    insights_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract the 5-7 most important insights from this paper...
         Format your response using the JSON schema:
        {
          "tool_calls": [
            {
              "name": "insights_result",
              "args": {
                "insights": [
                  {
                    "headline": "Main insight 1",
                    "description": "Detailed explanation 1"
                  },
                  {
                    "headline": "Main insight 2", 
                    "description": "Detailed explanation 2"
                  }
                  // ... more insights
                ]
              }
            }
          ]
        }
         """),
        ("user", "Title: {title}\nAbstract: {abstract}\nContent: {paper_content}")
    ])
    
    parser = JsonOutputToolsParser()
    chain = insights_prompt | creative_llm | parser
    
    try:
        insights = chain.invoke({
            "title": metadata.get("title", "Unknown"),
            "abstract": metadata.get("abstract", "No abstract"),
            "paper_content": paper_content
        })
        return {**state, "key_insights": insights, "current_step": "identify_key_insights"}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "error"}

# Other functions like `conduct_background_research`, `create_podcast_outline`, etc., follow the same pattern.


def conduct_background_research(state: PodcastState) -> PodcastState:
    """Conduct additional background research to add context to the podcast."""
    metadata = state["paper_metadata"]
    insights = state["key_insights"]
    
    # Create search queries based on paper title and key insights
    title = metadata.get("title", "")
    authors = metadata.get("authors", [])
    
    search_queries = [
        f"latest research on {title}",
        f"background context for {title}",
    ]
    
    # Add author-specific queries
    if authors:
        if isinstance(authors, list) and len(authors) > 0:
            main_author = authors[0]
            search_queries.append(f"other research by {main_author}")
        elif isinstance(authors, str):
            search_queries.append(f"other research by {authors}")
    
    # Add insight-based queries
    for i, insight in enumerate(insights[:2]):  # Use first two insights for queries
        if isinstance(insight, dict) and "headline" in insight:
            search_queries.append(f"research on {insight['headline']}")
    
    # Conduct searches
    search_results = []
    for query in search_queries:
        try:
            results = search_tool.invoke({"query": query, "max_results": 3})
            search_results.extend(results)
        except Exception as e:
            print(f"Search error for query '{query}': {str(e)}")
    
    # Process and consolidate research notes
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research producer for an academic podcast.
        Based on the paper metadata and search results provided, create comprehensive
        research notes that can provide valuable context for a podcast discussion.
        
        Include:
        1. Background on the broader field
        2. Related work and how it connects
        3. Potential real-world applications
        4. Controversies or debates in the field
        5. Future research directions
        
        Write this as a structured document with clear sections."""),
        ("user", """Paper Metadata:
        {metadata}
        
        Key Insights:
        {insights}
        
        Search Results:
        {search_results}""")
    ])
    
    chain = research_prompt | creative_llm | OutputFixingParser()
    
    try:
        start_time = datetime.now(timezone.utc)
        research_notes = chain.invoke({
            "metadata": metadata,
            "insights": insights,
            "search_results": search_results
        })
        
        # Save research notes to file
        save_to_file(research_notes, "research_notes.md")
        
        agentops.log_event(
            event_type="background_research",
            properties={
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "search_queries": search_queries,
                "result_count": len(search_results)
            }
        )
        
        return {
            **state,
            "research_notes": research_notes,
            "current_step": "conduct_background_research"
        }
    except Exception as e:
        error_msg = f"Error conducting background research: {str(e)}"
        agentops.log_event(
            event_type="background_research_error",
            properties={"error": str(e), "traceback": traceback.format_exc()}
        )
        return {**state, "error": error_msg, "current_step": "error"}



def create_podcast_outline(state: PodcastState) -> PodcastState:
    """Create an outline for the podcast episode."""
    metadata = state["paper_metadata"]
    insights = state["key_insights"]
    research_notes = state["research_notes"]
    
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert podcast producer who specializes in creating
        engaging and educational podcast episodes about academic research.
        
        Create a detailed podcast outline for a 20-30 minute episode about this research paper.
        Include:
        1. An attention-grabbing cold open
        2. Brief intro to the hosts (two hosts - an expert and a curious generalist)
        3. Introduction to the paper and authors
        4. 3-5 main segments that explore the key insights
        5. Discussion of implications and applications
        6. Closing thoughts and call to action
        
        For each segment, include:
        - Segment title
        - Key talking points
        - Questions for discussion
        - Estimated duration
        
        Format your response as a structured JSON object with appropriate sections."""),
        ("user", """Paper Information:
        {metadata}
        
        Key Insights:
        {insights}
        
        Research Notes:
        {research_notes}""")
    ])
    
    parser = JsonOutputToolsParser()
    chain = outline_prompt | creative_llm | parser
    
    try:
        start_time = datetime.now(timezone.utc)
        outline = chain.invoke({
            "metadata": metadata,
            "insights": insights,
            "research_notes": research_notes
        })
        
        # Save outline to file
        save_to_file(str(outline), "podcast_outline.json")
        
        agentops.log_event(
            event_type="podcast_outline_creation",
            properties={
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "segment_count": len(outline.get("segments", []))
            }
        )
        
        return {
            **state,
            "podcast_outline": outline,
            "current_step": "create_podcast_outline"
        }
    except Exception as e:
        error_msg = f"Error creating podcast outline: {str(e)}"
        agentops.log_event(
            event_type="podcast_outline_error",
            properties={"error": str(e), "traceback": traceback.format_exc()}
        )
        return {**state, "error": error_msg, "current_step": "error"}
    


def write_podcast_script(state: PodcastState) -> PodcastState:
    """Write the full podcast script based on the outline."""
    metadata = state["paper_metadata"]
    outline = state["podcast_outline"]
    insights = state["key_insights"]
    research_notes = state["research_notes"]
    
    script_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert podcast script writer who specializes in
        creating engaging, educational, and conversational scripts about complex research topics.
        
        Write a full podcast script based on the outline provided. The podcast has two hosts:
        - Dr. Alex: The expert host who understands the technical details
        - Jamie: The curious co-host who asks questions the audience might have
        
        Make the conversation sound natural and engaging, while being educational.
        Include:
        - Sound cues and production notes in [brackets]
        - Speaker labels (DR. ALEX: and JAMIE:)
        - Natural transitions between segments
        - Moments of humor and personality
        - Clear explanations of complex concepts
        - Questions that probe deeper into the research
        
        Format your response as a properly formatted podcast script with clear speaker 
        attributions and production notes."""),
        ("user", """Paper Information:
        {metadata}
        
        Podcast Outline:
        {outline}
        
        Key Insights:
        {insights}
        
        Research Notes:
        {research_notes}""")
    ])
    
    parser = JsonOutputToolsParser()
    chain = script_prompt | creative_llm | parser
    
    try:
        start_time = datetime.now(timezone.utc)
        script = chain.invoke({
            "metadata": metadata,
            "outline": outline,
            "insights": insights,
            "research_notes": research_notes
        })
        
        # Save script to file
        script_text = script.get("full_script", "")
        save_to_file(script_text, "podcast_script.txt")
        
        agentops.log_event(
            event_type="podcast_script_creation",
            properties={
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "script_length": len(script_text),
                "estimated_duration_minutes": script.get("estimated_duration_minutes", 0)
            }
        )
        
        return {
            **state,
            "podcast_script": script,
            "current_step": "write_podcast_script"
        }
    except Exception as e:
        error_msg = f"Error writing podcast script: {str(e)}"
        agentops.log_event(
            event_type="podcast_script_error",
            properties={"error": str(e), "traceback": traceback.format_exc()}
        )
        return {**state, "error": error_msg, "current_step": "error"}

def analyze_script(state: PodcastState) -> PodcastState:
    """Analyze the script for quality and areas of improvement."""
    script = state["podcast_script"]
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert podcast producer and editor.
        Analyze the podcast script for quality, engagement, clarity, and educational value.
        
        Provide:
        1. Overall quality assessment (1-10 scale)
        2. Strengths of the script
        3. Areas for improvement
        4. Specific edit suggestions for 2-3 sections
        5. Recommendations for sound design or music
        6. Estimated audience engagement level
        
        Be honest but constructive in your feedback."""),
        ("user", "Podcast Script:\n{script}")
    ])
    
    parser = JsonOutputToolsParser()
    chain = analysis_prompt | creative_llm | parser
    
    try:
        start_time = datetime.now(timezone.utc)
        analysis = chain.invoke({
            "script": script.get("full_script", "")
        })
        
        # Save analysis to file
        save_to_file(str(analysis), "script_analysis.json")
        
        agentops.log_event(
            event_type="script_analysis",
            properties={
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "quality_score": analysis.get("quality_score", 0)
            }
        )
        
        return {
            **state,
            "podcast_analysis": analysis,
            "current_step": "analyze_script"
        }
    except Exception as e:
        error_msg = f"Error analyzing script: {str(e)}"
        agentops.log_event(
            event_type="script_analysis_error",
            properties={"error": str(e), "traceback": traceback.format_exc()}
        )
        return {**state, "error": error_msg, "current_step": "error"}

def create_final_output(state: PodcastState) -> PodcastState:
    """Create the final output package with all podcast production materials."""
    metadata = state["paper_metadata"]
    script = state["podcast_script"]
    analysis = state["podcast_analysis"]
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a podcast production manager responsible for creating
        the final production package for an academic podcast episode.
        
        Create a comprehensive production package that includes:
        1. Episode title and subtitle
        2. Episode description (short and long versions)
        3. Show notes with timestamps and references
        4. Social media promotional text (3 versions - Twitter, LinkedIn, Instagram)
        5. Key quotes for promotion
        6. SEO keywords and tags
        7. Production timeline and checklist
        
        Format your response as a well-structured JSON object with all these components."""),
        ("user", """Paper Information:
        {metadata}
        
        Podcast Script:
        {script}
        
        Script Analysis:
        {analysis}""")
    ])
    
    parser = JsonOutputToolsParser()
    chain = final_prompt | creative_llm | parser
    
    try:
        start_time = datetime.now(timezone.utc)
        final_output = chain.invoke({
            "metadata": metadata,
            "script": script,
            "analysis": analysis
        })
        
        # Save final output to file
        save_to_file(str(final_output), "production_package.json")
        
        # Create a markdown summary file
        summary = f"""# Research Paper to Podcast - Production Package

## Episode Information
- **Title**: {final_output.get('episode_title', 'Untitled Episode')}
- **Subtitle**: {final_output.get('episode_subtitle', '')}
- **Based on**: {metadata.get('title', 'Unknown Paper')}
- **Authors**: {metadata.get('authors', 'Unknown Authors')}

## Episode Description
{final_output.get('long_description', 'No description available')}

## Production Materials Available
- Full research notes
- Podcast outline 
- Complete script ({script.get('estimated_duration_minutes', '20-30')} minutes)
- Show notes with references
- Social media promotional content
- Production timeline

## Next Steps
{final_output.get('next_steps', '1. Review and approve script\n2. Schedule recording session\n3. Record and edit episode\n4. Publish and promote')}
"""
        save_to_file(summary, "podcast_summary.md")
        
        agentops.log_event(
            event_type="final_output_creation",
            properties={
                "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "title": final_output.get('episode_title', '')
            }
        )
        
        return {
            **state,
            "final_output": final_output,
            "current_step": "create_final_output"
        }
    except Exception as e:
        error_msg = f"Error creating final output: {str(e)}"
        agentops.log_event(
            event_type="final_output_error",
            properties={"error": str(e), "traceback": traceback.format_exc()}
        )
        return {**state, "error": error_msg, "current_step": "error"}



def handle_error(state: PodcastState) -> PodcastState:
    """Handle errors in workflow execution."""
    error_report = f"Error in step {state['current_step']}: {state['error']}"
    save_to_file(error_report, "error_report.txt")
    return {**state, "current_step": "error_handled"}

# -----------------------------
# 🌟 LangGraph Workflow Setup 🌟
# -----------------------------

workflow = StateGraph(PodcastState)

# ✅ Add nodes
workflow.add_node("extract_metadata", extract_metadata)
workflow.add_node("identify_key_insights", identify_key_insights)
workflow.add_node("conduct_background_research", conduct_background_research)
workflow.add_node("create_podcast_outline", create_podcast_outline)
workflow.add_node("write_podcast_script", write_podcast_script)
workflow.add_node("analyze_script", analyze_script)
workflow.add_node("create_final_output", create_final_output)
workflow.add_node("handle_error", handle_error)

# ✅ Define execution order
workflow.add_edge("extract_metadata", "identify_key_insights")
workflow.add_edge("identify_key_insights", "conduct_background_research")
workflow.add_edge("conduct_background_research", "create_podcast_outline")
workflow.add_edge("create_podcast_outline", "write_podcast_script")
workflow.add_edge("write_podcast_script", "analyze_script")
workflow.add_edge("analyze_script", "create_final_output")

# ✅ Error Handling (Redirect failures to error handler)
workflow.add_conditional_edges(
    "extract_metadata",
    lambda state: "error" in state,
    {"True": "handle_error", "False": "identify_key_insights"},
)
workflow.add_conditional_edges(
    "identify_key_insights",
    lambda state: "error" in state,
    {"True": "handle_error", "False": "conduct_background_research"},
)
workflow.add_conditional_edges(
    "write_podcast_script",
    lambda state: "error" in state,
    {"True": "handle_error", "False": "analyze_script"},
)

# ✅ Set entry & exit points
workflow.set_entry_point("extract_metadata")

# ✅ Compile the workflow
app = workflow.compile()

# ✅ Run the workflow (example usage)
if __name__ == "__main__":
    initial_state = {"paper_content": read_pdf(r"D:\Data\Desktop\ICT\ai_final\p2p\knowledge\Video_Stream_Analysis_in_Clouds_An_Object_Detection_and_Classification_Framework_for_High_Performance_Video_Analytics.pdf"), "current_step": ""}
    result = app.invoke(initial_state)
    print("Workflow completed! 🎙️", result)

    # writing the result to a .txt_file
    with open("podcast_generation_result.txt", "w") as f:
        f.write(str(result))

