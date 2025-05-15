import streamlit as st
import os
import io
import json
import base64
import tempfile
from dotenv import load_dotenv
import openai
from PIL import Image
import pandas as pd
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta
import time
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import fitz  # PyMuPDF

# Set page configuration
st.set_page_config(
    page_title="Resume Parser App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Authentication
def setup_openai_api():
    if 'OPENAI_API_KEY' in st.session_state:
        return True

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.session_state['OPENAI_API_KEY'] = api_key
        return True

    # If no API key in environment, ask user
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        st.session_state['OPENAI_API_KEY'] = api_key
        return True
    return False

# Helper functions
def compress_image(image, max_size_mb=4):
    """Compress image to stay under max_size_mb"""
    img_byte_arr = io.BytesIO()
    quality = 95
    image.save(img_byte_arr, format='JPEG', quality=quality)

    while img_byte_arr.tell() > max_size_mb * 1024 * 1024 and quality > 10:
        img_byte_arr = io.BytesIO()
        quality -= 5
        image.save(img_byte_arr, format='JPEG', quality=quality)

    return img_byte_arr.getvalue()

def convert_pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to images using PyMuPDF"""
    try:
        # Calculate zoom factor based on DPI (default PDF is 72 DPI)
        zoom = dpi / 72

        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        images = []

        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Render page to an image with the specified zoom
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Compress the image
            img_bytes = compress_image(img)
            images.append(img_bytes)

        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF {os.path.basename(pdf_path)}: {str(e)}")
        return None

def encode_image(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def parse_date(date_str):
    """Parse date string to standard format with specific handling for recent dates"""
    if not date_str:
        return None

    # Handle "Present" or current dates
    if isinstance(date_str, str) and re.search(r'present|current|now', date_str.lower()):
        return datetime.now().strftime('%Y-%m-%d')

    # Print raw input for debugging
    print(f"Parsing date from: '{date_str}'")
    
    try:
        # Clean the date string
        date_str = date_str.strip()
        
        # SPECIFIC FIX FOR 12.2024 FORMAT
        # This pattern looks for month.year format like 12.2024
        special_format = re.search(r'(\d{1,2})\.(\d{4})', date_str)
        if special_format:
            month = int(special_format.group(1))
            year = int(special_format.group(2))
            
            # Validate month
            if 1 <= month <= 12:
                print(f"Found special format date: month={month}, year={year}")
                # Ensure recent years are correctly interpreted
                current_year = datetime.now().year
                if year > current_year - 5 and year <= current_year + 2:
                    return f"{year}-{month:02d}-01"
        
        # Extra check specifically for 2024/2004 confusion in OCR
        # First, check if the string contains "2024" explicitly
        if "2024" in date_str:
            print("Found '2024' explicitly in date string")
            month_match = re.search(r'(\d{1,2})[\/\.\-]2024', date_str)
            if month_match:
                month = int(month_match.group(1))
                if 1 <= month <= 12:
                    return f"2024-{month:02d}-01"
            return "2024-01-01"  # Default to January if month not found
            
        # Try standard date formats
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%b %Y', '%B %Y', '%Y', '%m/%Y', '%d-%m-%Y', '%m-%Y']:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                
                # Critical fix for the 2004/2024 confusion
                # If we parsed 2004 but the original text contains indicators of 2024
                if parsed_date.year == 2004 and ("202" in date_str or "24" in date_str):
                    print(f"Correcting likely OCR error: 2004 -> 2024")
                    return f"2024-{parsed_date.month:02d}-{parsed_date.day:02d}"
                
                # Sanity check for dates
                current_year = datetime.now().year
                if parsed_date.year < 1950 or parsed_date.year > current_year + 2:
                    print(f"Rejecting unlikely year: {parsed_date.year}")
                    continue
                
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # Aggressive pattern matching for years
        year_patterns = [
            (r'202[0-4]', lambda y: int(y)),  # Matches 2020-2024
            (r'20\s*2[0-4]', lambda y: int(y.replace(' ', ''))),  # Handles OCR spacing issues
        ]
        
        for pattern, year_parser in year_patterns:
            year_match = re.search(pattern, date_str)
            if year_match:
                try:
                    year = year_parser(year_match.group(0))
                    print(f"Found year with pattern matching: {year}")
                    
                    # Try to extract month
                    month = 1  # Default to January
                    month_patterns = [
                        r'(\d{1,2})[\/\.\-]',  # Matches 12.2024, 12/2024, 12-2024
                        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'  # Month names
                    ]
                    
                    for m_pattern in month_patterns:
                        m_match = re.search(m_pattern, date_str, re.IGNORECASE)
                        if m_match:
                            m_str = m_match.group(1)
                            if m_str.isdigit():
                                month = int(m_str)
                                if 1 <= month <= 12:
                                    break
                            else:
                                try:
                                    month = datetime.strptime(m_str, '%b').month
                                    break
                                except ValueError:
                                    pass
                    
                    return f"{year}-{month:02d}-01"
                except Exception as e:
                    print(f"Error processing year match: {e}")
        
        # Last resort: check for 4-digit numbers that could be years
        year_match = re.search(r'(20\d{2}|19\d{2})', date_str)
        if year_match:
            year = int(year_match.group(1))
            
            # Special case for 2004 which might actually be 2024
            if year == 2004 and ("Present" in date_str or datetime.now().year - year > 15):
                print("Converting suspected 2004 to 2024 based on context")
                year = 2024
                
            print(f"Extracted year as last resort: {year}")
            return f"{year}-01-01"
            
        print(f"Failed to parse date: '{date_str}'")
        return None
    except Exception as e:
        print(f"Error parsing date '{date_str}': {str(e)}")
        return None

def calculate_experience_duration(start_date, end_date):
    """Calculate duration between two dates in years and months"""
    if not start_date:
        return 0

    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Debug printing
        print(f"Calculating experience: {start_date} to {end_date}")

        if not end_date:
            end = datetime.now()
            print(f"No end date provided, using current date: {end}")
        else:
            end = datetime.strptime(end_date, '%Y-%m-%d')

        # Critical sanity check
        current_year = datetime.now().year
        if start.year < current_year - 60:
            print(f"Suspicious start year {start.year}, might be OCR error.")
            # Check if it might be mistaking century
            if start.year < 2000 and current_year - 100 < start.year < current_year - 70:
                print(f"Correcting century: {start.year} -> {start.year + 100}")
                start = start.replace(year=start.year + 100)
        
        # Another sanity check for 2004/2024 confusion
        if start.year == 2004 and end.year >= 2024:
            # This could be a case where 2024 was misread as 2004
            years_diff = end.year - start.year
            if years_diff > 15:  # Unusually long employment period
                print(f"Suspicious date range: {start.year} to {end.year}. Correcting 2004 to 2024.")
                start = start.replace(year=2024)

        if end < start:
            print(f"Warning: End date {end_date} is before start date {start_date}.")
            # If the difference is small, it might be a mistake - swap them
            if (start - end).days < 30:
                start, end = end, start
                print("Dates were close together - swapped them.")
            else:
                # If we have an end date of "Present" and start date in future,
                # the start date is likely wrong
                if end.year == datetime.now().year:
                    start = start.replace(year=start.year - 20)
                    print(f"Adjusted suspicious start date to {start.year}")

        diff = relativedelta(end, start)
        years_decimal = diff.years + (diff.months / 12) + (diff.days / 365.25)
        
        # Final sanity check
        if years_decimal > 30:
            print(f"Warning: Calculated experience ({years_decimal} years) seems too long.")
            # If suspicious, cap at reasonable amount
            years_decimal = min(years_decimal, 25)
            
        result = round(years_decimal, 2)
        print(f"Calculated experience duration: {result} years")
        return result
    except Exception as e:
        print(f"Error calculating experience: {str(e)}")
        return 0


# Patch function for extract_text_from_images to modify
def patch_extract_text_from_images(original_function):
    def wrapped_function(images, pdf_path):
        result = original_function(images, pdf_path)
        
        if result:
            # Introduce additional post-processing specifically for dates
            for exp in result.get('experience', []):
                # Fix known OCR issues with dates
                if exp.get('start_date') == '2004-01-01' and 'present' in str(exp.get('end_date')).lower():
                    print("Applying post-processing fix for suspected 2004/2024 confusion")
                    exp['start_date'] = '2024-01-01'
                
                # Check for unusually long employment periods
                start_date = exp.get('start_date')
                end_date = exp.get('end_date')
                
                if start_date and end_date:
                    try:
                        start = datetime.strptime(start_date, '%Y-%m-%d')
                        end = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        # If employment is over 15 years and starts with "200", check if it's 2024
                        if (end - start).days > 365 * 15 and str(start.year).startswith('200'):
                            corrected_year = 2024
                            print(f"Post-processing date correction: {start.year} -> {corrected_year}")
                            exp['start_date'] = f"{corrected_year}-{start.month:02d}-{start.day:02d}"
                    except Exception as e:
                        print(f"Error in post-processing dates: {e}")
        
        return result
    
    return wrapped_function

def extract_text_from_images(images, pdf_path):
    """Extract information from resume images using OpenAI Vision"""

    system_prompt = """You are a precise resume information extractor.
    Extract exactly the requested fields from the resume and return them in the specified JSON format.
    For arrays of objects (experience, education, etc.), ensure all entries are complete.
    If a field is not found, set it to null. Ensure all found values are strings."""

    user_prompt = """Extract the following fields from this resume and return them in exactly this JSON format:
    {
        "name": null,
        "email": null,
        "phone": null,
        "skills": [],
        "experience": [
            {
                "company_name": null,
                "job_title": null,
                "start_date": null,
                "end_date": null,
                "location": null,
                "employment_type": null,
                "responsibilities": null,
                "reference_contact_person": null,
                "reference_contact_number": null,
                "reference_designation": null,
                "comments_notes": null
            }
        ],
        "education": [
            {
                "degree_qualification": null,
                "field_of_study": null,
                "institute_name": null,
                "start_date": null,
                "end_date": null,
                "grade_gpa": null,
                "education_level": null,
                "location": null,
                "certification": null,
                "comments_notes": null
            }
        ],
        "location": {
            "location_name": null,
            "country": null,
            "state_province": null,
            "city": null
        },
        "languages": [
            {
                "language_name": null,
                "proficiency": null
            }
        ]
    }"""

    try:
        # Process first few pages only
        first_pages = images[:10]

        # Prepare messages for the API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        # Add images to the message
        for img_bytes in first_pages:
            base64_image = encode_image(img_bytes)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # Call OpenAI API
        openai.api_key = st.session_state['OPENAI_API_KEY']
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=4000,
            temperature=0
        )

        # Extract and parse JSON from response
        response_text = response.choices[0].message.content

        # Clean JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        # Parse and validate JSON
        extracted_info = json.loads(response_text.strip())

        # Calculate total experience
        total_experience = 0
        for exp in extracted_info.get('experience', []):
            start_date = parse_date(exp.get('start_date'))
            end_date = parse_date(exp.get('end_date'))
            exp['start_date'] = start_date
            exp['end_date'] = end_date

            # Calculate experience duration for this position
            duration = calculate_experience_duration(start_date, end_date)
            exp['duration_years'] = duration
            total_experience += duration

        # Add total experience to extracted info
        extracted_info['total_experience_years'] = round(total_experience, 2)

        # Process education dates
        for edu in extracted_info.get('education', []):
            edu['start_date'] = parse_date(edu.get('start_date'))
            edu['end_date'] = parse_date(edu.get('end_date'))

        # Add source file information
        extracted_info['source_file'] = os.path.basename(pdf_path)
        return extracted_info

    except Exception as e:
        st.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
        return None

def process_resume(pdf_path, progress_bar=None, progress_text=None):
    """Process a single resume PDF"""
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)
    if not images:
        return None

    # Extract information
    if progress_text:
        progress_text.text(f"Extracting data from {os.path.basename(pdf_path)}...")

    info = extract_text_from_images(images, pdf_path)

    # Update progress if needed
    if progress_bar:
        progress_bar.progress(1.0)

    return info

def process_batch(uploaded_files, progress_placeholder):
    """Process a batch of resume files with progress tracking"""
    results = []

    progress_bar = progress_placeholder.progress(0.0)
    progress_text = progress_placeholder.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        progress_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")

        # Save uploaded file with original name
        file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process individual file with dedicated progress
        file_progress = progress_placeholder.progress(0.0)
        file_progress_text = progress_placeholder.empty()

        # Process the file
        info = process_resume(file_path, file_progress, file_progress_text)
        if info:
            results.append(info)
            progress_text.success(f"✅ Successfully processed {uploaded_file.name}")
        else:
            progress_text.error(f"❌ Failed to process {uploaded_file.name}")

        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

        # Remove individual file progress after completion
        file_progress.empty()
        file_progress_text.empty()

        # Update overall progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)

    progress_text.text("Processing complete!")
    return results

def create_consolidated_df(results):
    """Create a consolidated DataFrame from results"""
    if not results:
        return None

    consolidated_data = []

    for result in results:
        source_file = result['source_file']
        name = result.get('name', '')
        email = result.get('email', '')
        phone = result.get('phone', '')
        total_experience = result.get('total_experience_years', 0)
        skills = ', '.join(result.get('skills', []))

        # Location details
        location = result.get('location', {})
        location_name = location.get('location_name', '')
        country = location.get('country', '')
        state_province = location.get('state_province', '')
        city = location.get('city', '')

        # Experience details - concatenate
        experience_details = []
        for exp in result.get('experience', []):
            exp_detail = f"{exp.get('job_title', 'N/A')} at {exp.get('company_name', 'N/A')}"
            if exp.get('start_date') and exp.get('end_date'):
                exp_detail += f" ({exp.get('start_date')} to {exp.get('end_date')})"
            experience_details.append(exp_detail)

        experience_summary = '; '.join(experience_details)

        # Education details - concatenate
        education_details = []
        for edu in result.get('education', []):
            edu_detail = f"{edu.get('degree_qualification', 'N/A')} in {edu.get('field_of_study', 'N/A')} from {edu.get('institute_name', 'N/A')}"
            education_details.append(edu_detail)

        education_summary = '; '.join(education_details)

        # Languages - concatenate
        language_details = []
        for lang in result.get('languages', []):
            lang_detail = f"{lang.get('language_name', 'N/A')} ({lang.get('proficiency', 'N/A')})"
            language_details.append(lang_detail)

        languages_summary = ', '.join(language_details)

        # Add to consolidated data
        consolidated_data.append({
            'Source File': source_file,
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Total Experience (Years)': total_experience,
            'Skills': skills,
            'Experience Summary': experience_summary,
            'Education Summary': education_summary,
            'Languages': languages_summary,
            'Location': f"{city}, {state_province}, {country}" if city or state_province or country else location_name
        })

    return pd.DataFrame(consolidated_data)

def create_entity_dataframes(results):
    """Create DataFrames for different entities in the resumes"""
    if not results:
        return {}

    experiences = []
    educations = []
    locations = []
    languages = []
    skills = []
    profiles = []

    for result in results:
        source_file = result['source_file']

        # Process experiences
        for exp in result.get('experience', []):
            exp['source_file'] = source_file
            exp['candidate_name'] = result.get('name', '')
            experiences.append(exp)

        # Process education
        for edu in result.get('education', []):
            edu['source_file'] = source_file
            edu['candidate_name'] = result.get('name', '')
            educations.append(edu)

        # Process location
        loc = result.get('location', {})
        loc['source_file'] = source_file
        loc['candidate_name'] = result.get('name', '')
        locations.append(loc)

        # Process languages
        for lang in result.get('languages', []):
            lang['source_file'] = source_file
            lang['candidate_name'] = result.get('name', '')
            languages.append(lang)

        # Process skills
        for skill in result.get('skills', []):
            skills.append({
                'source_file': source_file,
                'candidate_name': result.get('name', ''),
                'skill': skill
            })

        # Add profile summary with total experience
        profiles.append({
            'source_file': source_file,
            'name': result.get('name'),
            'email': result.get('email'),
            'phone': result.get('phone'),
            'total_experience_years': result.get('total_experience_years', 0)
        })

    # Create DataFrames
    dfs = {}
    if experiences:
        dfs['experiences'] = pd.DataFrame(experiences)
    if educations:
        dfs['educations'] = pd.DataFrame(educations)
    if locations:
        dfs['locations'] = pd.DataFrame(locations)
    if languages:
        dfs['languages'] = pd.DataFrame(languages)
    if skills:
        dfs['skills'] = pd.DataFrame(skills)
    if profiles:
        dfs['profiles'] = pd.DataFrame(profiles)

    return dfs

def create_visualizations(consolidated_df, entity_dfs):
    """Create visualizations from the extracted data"""
    visualizations = {}

    if consolidated_df is not None and not consolidated_df.empty:
        # 2. Common skills
        if 'skills' in entity_dfs and not entity_dfs['skills'].empty:
            skill_counts = entity_dfs['skills']['skill'].value_counts().head(15)
            fig2 = px.bar(
                x=skill_counts.values,
                y=skill_counts.index,
                orientation='h',
                title='Most Common Skills',
                labels={'x': 'Count', 'y': 'Skill'},
                color_discrete_sequence=['#f72585']
            )
            visualizations['skills_bar'] = fig2

        # 3. Education levels
        if 'educations' in entity_dfs and not entity_dfs['educations'].empty:
            edu_df = entity_dfs['educations']
            if 'education_level' in edu_df.columns:
                edu_counts = edu_df['education_level'].value_counts()
                fig3 = px.pie(
                    values=edu_counts.values,
                    names=edu_counts.index,
                    title='Education Levels',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                visualizations['education_pie'] = fig3

    return visualizations

def create_download_files(consolidated_df, entity_dfs):
    """Create download files (zip containing all CSVs)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save consolidated CSV
        consolidated_path = os.path.join(tmpdir, 'consolidated_resumes.csv')
        consolidated_df.to_csv(consolidated_path, index=False)

        # Save entity CSVs
        for name, df in entity_dfs.items():
            entity_path = os.path.join(tmpdir, f'extracted_{name}.csv')
            df.to_csv(entity_path, index=False)

        # Create ZIP file
        zip_path = os.path.join(tmpdir, 'resume_analysis_results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(consolidated_path, os.path.basename(consolidated_path))
            for name in entity_dfs:
                entity_path = os.path.join(tmpdir, f'extracted_{name}.csv')
                zipf.write(entity_path, os.path.basename(entity_path))

        # Read ZIP file
        with open(zip_path, 'rb') as f:
            return f.read()

def display_resume_details(result):
    """Display details of a single resume in an expandable format"""
    name = result.get('name', 'Unknown')
    email = result.get('email', 'N/A')
    phone = result.get('phone', 'N/A')
    total_exp = result.get('total_experience_years', 0)

    # Basic info
    st.subheader(name)
    cols = st.columns(4)
    cols[0].metric("Experience", f"{total_exp} years")
    cols[1].write(f"📧 {email}")
    cols[2].write(f"📞 {phone}")

    # Location
    location = result.get('location', {})
    loc_str = ", ".join([v for k, v in location.items() if v and k != 'location_name'])
    if not loc_str and location.get('location_name'):
        loc_str = location.get('location_name')
    cols[3].write(f"📍 {loc_str if loc_str else 'N/A'}")

    # Skills
    st.write("#### Skills")
    skills = result.get('skills', [])
    if skills:
        st.write(", ".join(skills))
    else:
        st.write("No skills listed")

    # Experience
    experience = result.get('experience', [])
    if experience:
        with st.expander("Experience", expanded=True):
            for idx, exp in enumerate(experience):
                st.markdown(f"**{exp.get('job_title', 'N/A')} at {exp.get('company_name', 'N/A')}**")

                # Dates and duration
                date_str = ""
                if exp.get('start_date'):
                    date_str += exp.get('start_date')
                if exp.get('end_date'):
                    date_str += f" to {exp.get('end_date')}"
                if date_str:
                    st.write(f"*{date_str}* ({exp.get('duration_years', 0)} years)")

                # Location and employment type
                if exp.get('location') or exp.get('employment_type'):
                    loc_emp = []
                    if exp.get('location'):
                        loc_emp.append(f"📍 {exp.get('location')}")
                    if exp.get('employment_type'):
                        loc_emp.append(exp.get('employment_type'))
                    st.write(" | ".join(loc_emp))

                # Responsibilities
                if exp.get('responsibilities'):
                    st.write(exp.get('responsibilities'))

                if idx < len(experience) - 1:
                    st.divider()

    # Education
    education = result.get('education', [])
    if education:
        with st.expander("Education", expanded=True):
            for idx, edu in enumerate(education):
                st.markdown(f"**{edu.get('degree_qualification', 'N/A')}** in *{edu.get('field_of_study', 'N/A')}*")
                st.write(f"📚 {edu.get('institute_name', 'N/A')}")

                # Dates
                date_str = ""
                if edu.get('start_date'):
                    date_str += edu.get('start_date')
                if edu.get('end_date'):
                    date_str += f" to {edu.get('end_date')}"
                if date_str:
                    st.write(f"*{date_str}*")

                # Grade/GPA
                if edu.get('grade_gpa'):
                    st.write(f"🎓 Grade/GPA: {edu.get('grade_gpa')}")

                if idx < len(education) - 1:
                    st.divider()

    # Languages
    languages = result.get('languages', [])
    if languages:
        with st.expander("Languages"):
            for lang in languages:
                st.write(f"**{lang.get('language_name', 'N/A')}**: {lang.get('proficiency', 'N/A')}")

def extract_jd_info(jd_text):
    """
    Extracts years of experience, skills, location, job type, and ideal candidate profile from a given job description using GPT-4.
    """
    prompt = f"""Extract the following details from the job description and provide the output strictly in the following JSON format:
    {{
        "experience": int,  # Total years of experience
        "skills": ["skill1", "skill2", "skill3"],  # List of required skills
        "location": "location_string",  # Location of the job
        "job_type": "onsite" or "remote",  # If not specified, assume "onsite"
        "ideal_candidate": ["requirement1", "requirement2", "requirement3", "requirement4", "requirement5"]  # Up to 5 specific requirements for the ideal candidate directly from the JD
    }}

    For the ideal_candidate field, extract up to 5 specific points directly from the job description that represent the key requirements. These should be the most important qualifications or experiences mentioned (like "2+ years experience in Python", "Bachelor's degree in Computer Science", etc.).

    Job Description:
    {jd_text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    # Remove triple backticks if present
    content = re.sub(r'^```json|```$', '', content).strip()

    # Validate JSON response
    try:
        extracted_data = json.loads(content)
    except json.JSONDecodeError as e:
        print("Error: The response is not valid JSON.", e)
        return None

    return extracted_data

def format_main_df(extracted_info):
    """
    Creates the main DataFrame with job details, formatting the ideal candidate as a string
    """
    if not extracted_info:
        return None

    # Format ideal candidate list as bulleted points
    if isinstance(extracted_info['ideal_candidate'], list):
        ideal_candidate = "\n• " + "\n• ".join(extracted_info['ideal_candidate'])
    else:
        ideal_candidate = str(extracted_info['ideal_candidate'])

    return pd.DataFrame([{
        'experience': extracted_info['experience'],
        'skills': ', '.join(extracted_info['skills']),
        'location': extracted_info['location'],
        'job_type': extracted_info['job_type'],
        'ideal_candidate': ideal_candidate
    }])

def compare_jd_with_resumes(jd_info, results):
    """
    Compare JD with resumes and rate/recommend candidates.
    Prioritize experience and keyword matching over location.
    """
    if not jd_info or not results:
        return None

    jd_skills = set(jd_info['skills'])
    jd_experience = jd_info['experience']
    jd_location = jd_info['location'].lower()

    recommendations = []

    for result in results:
        candidate_name = result.get('name', 'Unknown')
        candidate_skills = set(result.get('skills', []))
        candidate_experience = result.get('total_experience_years', 0)
        candidate_location = result.get('location', {}).get('location_name', '').lower()

        # Calculate skill match score
        skill_match = len(jd_skills.intersection(candidate_skills)) / len(jd_skills) if jd_skills else 0

        # Calculate experience match score
        experience_match = min(candidate_experience / jd_experience, 1) if jd_experience else 0

        # Calculate location match score
        location_match = 1 if jd_location in candidate_location else 0

        # Calculate overall match score with adjusted weights
        overall_match = (skill_match * 0.5) + (experience_match * 0.4) + (location_match * 0.1)

        recommendations.append({
            'name': candidate_name,
            'skill_match': skill_match,
            'experience_match': experience_match,
            'location_match': location_match,
            'overall_match': overall_match
        })

    # Sort recommendations by overall match score
    recommendations.sort(key=lambda x: x['overall_match'], reverse=True)

    return pd.DataFrame(recommendations)

# Main Streamlit App
def main():
    # App title and introduction
    st.title("🔍 Resume Parser")
    st.write("""
    Upload multiple resumes (PDF format) to extract key information using AI.
    This tool will analyze skills, experience, education, and more.
    """)

    # Check for API key
    if not setup_openai_api():
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use this app.")
        return

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'consolidated_df' not in st.session_state:
        st.session_state['consolidated_df'] = None
    if 'entity_dfs' not in st.session_state:
        st.session_state['entity_dfs'] = {}
    if 'visualizations' not in st.session_state:
        st.session_state['visualizations'] = {}
    if 'jd_info' not in st.session_state:
        st.session_state['jd_info'] = None

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Batch size slider
        batch_size = st.slider("Maximum PDFs to Process", min_value=1, max_value=100, value=20,
                              help="Increase if you need to process more resumes at once")

        # Reset button
        if st.button("Reset App"):
            st.session_state['results'] = None
            st.session_state['consolidated_df'] = None
            st.session_state['entity_dfs'] = {}
            st.session_state['visualizations'] = {}
            st.session_state['jd_info'] = None
            st.rerun()

    # File uploader
    uploaded_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

    # Process files
    if uploaded_files:
        if len(uploaded_files) > batch_size:
            st.warning(f"⚠️ You uploaded {len(uploaded_files)} files, but the maximum is set to {batch_size}. Only the first {batch_size} files will be processed.")
            uploaded_files = uploaded_files[:batch_size]

        if st.button(f"Process {len(uploaded_files)} Resume{'s' if len(uploaded_files) > 1 else ''}"):
            progress_placeholder = st.empty()

            with st.spinner("Processing resumes..."):
                # Process the batch
                results = process_batch(uploaded_files, progress_placeholder)

                if results:
                    st.session_state['results'] = results
                    st.session_state['consolidated_df'] = create_consolidated_df(results)
                    st.session_state['entity_dfs'] = create_entity_dataframes(results)
                    st.session_state['visualizations'] = create_visualizations(
                        st.session_state['consolidated_df'],
                        st.session_state['entity_dfs']
                    )
                    st.success(f"✅ Successfully processed {len(results)} out of {len(uploaded_files)} resumes!")
                    st.rerun()
                else:
                    st.error("❌ No resumes were successfully processed.")

    # Display results if available
    if st.session_state['results']:
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👤 Individual Resumes", "📈 Analysis", "📁 Data Export", "📄 JD Parsing"])

        # Tab 1: Overview
        with tab1:
            st.header("Resume Processing Results")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Resumes", len(st.session_state['results']))

            # Average experience
            if 'consolidated_df' in st.session_state and st.session_state['consolidated_df'] is not None:
                avg_exp = st.session_state['consolidated_df']['Total Experience (Years)'].mean()
                col2.metric("Average Experience", f"{avg_exp:.1f} years")

                # Most common skills if available
                if 'skills' in st.session_state['entity_dfs'] and not st.session_state['entity_dfs']['skills'].empty:
                    top_skill = st.session_state['entity_dfs']['skills']['skill'].value_counts().index[0]
                    col3.metric("Top Skill", top_skill)

            # Display consolidated table
            if st.session_state['consolidated_df'] is not None:
                st.subheader("All Candidates")
                st.dataframe(st.session_state['consolidated_df'], use_container_width=True)

            # Display visualizations
            if 'skills_bar' in st.session_state['visualizations']:
                st.plotly_chart(st.session_state['visualizations']['skills_bar'], use_container_width=True)

        # Tab 2: Individual Resumes
        with tab2:
            st.header("Individual Resume Details")

            # Selector for individual resumes
            if st.session_state['results']:
                # Create a list of names for selection
                names = [f"{res.get('name', 'Unknown')} ({res['source_file']})" for res in st.session_state['results']]
                selected_name = st.selectbox("Select a resume to view", names)

                # Get the selected resume
                selected_index = names.index(selected_name)
                selected_resume = st.session_state['results'][selected_index]

                # Display the resume details
                st.divider()
                display_resume_details(selected_resume)

        # Tab 3: Analysis
        with tab3:
            st.header("Resume Analysis")

            # Skills analysis
            st.subheader("Skills Analysis")
            if 'skills_bar' in st.session_state['visualizations']:
                st.plotly_chart(st.session_state['visualizations']['skills_bar'], use_container_width=True)
            else:
                st.info("Not enough skill data to generate visualization")

            # Experience filtering
            st.subheader("Experience Filtering")
            if 'consolidated_df' in st.session_state and st.session_state['consolidated_df'] is not None:
                min_exp, max_exp = st.slider(
                    "Filter by years of experience",
                    0.0,
                    max(st.session_state['consolidated_df']['Total Experience (Years)'].max(), 20.0),
                    (0.0, 20.0)
                )

                filtered_df = st.session_state['consolidated_df'][
                    (st.session_state['consolidated_df']['Total Experience (Years)'] >= min_exp) &
                    (st.session_state['consolidated_df']['Total Experience (Years)'] <= max_exp)
                ]

                st.write(f"Found {len(filtered_df)} candidates with {min_exp}-{max_exp} years of experience")
                st.dataframe(filtered_df, use_container_width=True)

        # Tab 4: Data Export
        with tab4:
            st.header("Export Data")

            # Create download button for all data
            if 'consolidated_df' in st.session_state and st.session_state['consolidated_df'] is not None:
                st.subheader("Download All Data")

                # Create ZIP file with all CSVs
                zip_data = create_download_files(
                    st.session_state['consolidated_df'],
                    st.session_state['entity_dfs']
                )

                st.download_button(
                    label="Download All Data (ZIP)",
                    data=zip_data,
                    file_name="resume_analysis_results.zip",
                    mime="application/zip"
                )

                # Individual tables for download
                st.subheader("Download Individual Tables")

                # Consolidated table
                csv_consolidated = st.session_state['consolidated_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Consolidated Data",
                    data=csv_consolidated,
                    file_name="consolidated_resumes.csv",
                    mime="text/csv",
                    key="download_consolidated"
                )

                # Entity tables
                for name, df in st.session_state['entity_dfs'].items():
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {name.capitalize()} Data",
                        data=csv_data,
                        file_name=f"extracted_{name}.csv",
                        mime="text/csv",
                        key=f"download_{name}"
                    )

        # Tab 5: JD Parsing
        with tab5:
            st.header("Job Description Parsing")

            # Text area for JD input
            jd_text = st.text_area("Enter the Job Description", height=300)

            # Button to parse JD
            if st.button("Parse Job Description"):
                if jd_text:
                    with st.spinner("Parsing Job Description..."):
                        jd_info = extract_jd_info(jd_text)
                        if jd_info:
                            st.session_state['jd_info'] = jd_info
                            st.success("✅ Successfully parsed Job Description!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to parse Job Description.")
                else:
                    st.warning("Please enter a Job Description.")

            # Display JD info if available
            if st.session_state['jd_info']:
                jd_df = format_main_df(st.session_state['jd_info'])
                st.subheader("Job Description Details")
                st.dataframe(jd_df, use_container_width=True)

                # Button to compare with resumes
                if st.button("Compare with Resumes"):
                    if st.session_state['results']:
                        with st.spinner("Comparing Job Description with Resumes..."):
                            recommendations_df = compare_jd_with_resumes(st.session_state['jd_info'], st.session_state['results'])
                            st.subheader("Candidate Recommendations")
                            st.dataframe(recommendations_df, use_container_width=True)
                    else:
                        st.warning("Please process resumes first.")

if __name__ == "__main__":
    main()
