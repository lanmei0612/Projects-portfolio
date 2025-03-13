import logging
import pandas as pd
from openai import OpenAI
from typing import Dict, List
import json
import PyPDF2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CareerPathAdvisor:
    def __init__(self, api_key: str):
        """Initialize the OpenAI API client"""
        self.client = OpenAI(api_key=api_key)
        # Model configuration
        self.model = "gpt-4o"
        self.temperature = 0.1
        self.max_tokens = 2048

    def get_structured_response(self, prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system",
                     "content": "You are an expert career advisor. Provide responses in the exact JSON format requested."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            logging.info(f"Raw API response: {response_text[:500]}...")

            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1]
                else:
                    json_str = response_text

                json_str = json_str.strip()
                return json.loads(json_str)

            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                logging.error(f"Attempted to parse: {json_str}")
                raise

        except Exception as e:
            logging.error(f"API error: {str(e)}")
            raise

    def analyze_resume_for_jobs(self, resume_text: str, jobs_df: pd.DataFrame) -> Dict:
        try:
            logging.info("Starting resume analysis...")

            # Format jobs for the prompt (limit to first 5 for testing)
            jobs_text = "\n\n".join([
                f"Index {idx}:\nTitle: {row['job_title']}\n"
                f"Company: {row['company_name']}\n"
                f"Location: {row['location']}\n"
                f"Technical Skills Required: {str(row['technical_skills'])}\n"
                f"Professional Skills Required: {str(row['professional_skills'])}\n"
                f"Minimum Qualifications: {str(row['minimum_qualifications'])}"
                for idx, row in jobs_df.head(5).iterrows()
            ])

            prompt = f"""You are an expert career advisor. Analyze this resume and find the best matching jobs.
            For each job, determine the match level based on skills and qualifications alignment.

            Provide your analysis in this exact JSON format:
            {{
                "best_matches": [
                    {{
                        "job_index": <number 0-4>,
                        "match_level": "LOW/MEDIUM/HIGH",
                        "key_matching_skills": {{
                            "technical": ["skill1", "skill2"],
                            "professional": ["skill1", "skill2"]
                        }},
                        "missing_skills": {{
                            "technical": ["skill1", "skill2"],
                            "professional": ["skill1", "skill2"]
                        }},
                        "location_match": "yes/no",
                        "qualification_match": "yes/no",
                        "reasoning": "brief explanation of the match"
                    }}
                ]
            }}

            Match Level Criteria:
            - HIGH: Strong match in both technical and professional skills (>70% match)
            - MEDIUM: Good match in either technical or professional skills (40-70% match)
            - LOW: Limited match in both areas (<40% match)

            YOU MUST RETURN AT LEAST ONE MATCH, even if the match level is LOW.

            Resume:
            {resume_text}

            Available Jobs:
            {jobs_text}

            Return ONLY the JSON structure above, no other text."""

            response = self.get_structured_response(prompt)

            # Ensure we have at least one match
            if not response.get('best_matches'):
                logging.warning("No matches found, providing default match")
                response['best_matches'] = [{
                    "job_index": 0,
                    "match_level": "LOW",
                    "key_matching_skills": {
                        "technical": ["basic skills"],
                        "professional": ["general skills"]
                    },
                    "missing_skills": {
                        "technical": ["specific skills needed"],
                        "professional": ["specific professional skills needed"]
                    },
                    "location_match": "no",
                    "qualification_match": "no",
                    "reasoning": "Default match provided due to no strong matches found"
                }]

            logging.info(f"Found {len(response['best_matches'])} matches")
            return response

        except Exception as e:
            logging.error(f"Resume analysis error: {str(e)}")
            return {
                "best_matches": [{
                    "job_index": 0,
                    "match_level": "LOW",
                    "key_matching_skills": {
                        "technical": ["error processing"],
                        "professional": ["error processing"]
                    },
                    "missing_skills": {
                        "technical": ["error processing"],
                        "professional": ["error processing"]
                    },
                    "location_match": "no",
                    "qualification_match": "no",
                    "reasoning": f"Error analyzing resume: {str(e)}"
                }]
            }

    def process_job_application(self, resume_text: str, jobs_df: pd.DataFrame, courses_df: pd.DataFrame) -> Dict:
        try:
            # 1. Analyze resume against jobs
            job_matches = self.analyze_resume_for_jobs(resume_text, jobs_df)
            logging.info("Completed resume analysis")

            if not job_matches or 'best_matches' not in job_matches or not job_matches['best_matches']:
                raise ValueError("No job matches found")

            # Select the best match based on match level using a scoring system
            match_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            best_match = max(job_matches['best_matches'], key=lambda m: match_scores.get(m['match_level'].upper(), 0))
            best_job = jobs_df.iloc[best_match['job_index']]
            logging.info(f"Found best match: {best_job['job_title']}")

            # 2. Get relevant courses
            technical_skills = best_match['missing_skills']['technical']
            if technical_skills and technical_skills[0] != "error processing":
                prompt = f"""Find relevant courses for these technical skills: {', '.join(technical_skills)}
                Return in this JSON format:
                {{
                    "recommendations": [
                        {{
                            "skill": "skill name",
                            "course_name": "course title",
                            "relevance_level": "LOW/MEDIUM/HIGH",
                            "description": "why this course"
                        }}
                    ]
                }}

                Use these criteria for relevance_level:
                - HIGH: Course directly teaches the required skill
                - MEDIUM: Course covers the skill as part of broader content
                - LOW: Course teaches related concepts

                YOU MUST RETURN AT LEAST ONE COURSE RECOMMENDATION.
                Return ONLY the JSON, no other text."""

                course_recommendations = self.get_structured_response(prompt)
                logging.info("Generated course recommendations")
            else:
                course_recommendations = {
                    "recommendations": [{
                        "skill": "error occurred",
                        "course_name": "No specific course recommended",
                        "relevance_level": "LOW",
                        "description": "Unable to process course recommendations"
                    }]
                }

            # 3. Generate development plan
            development_prompt = f"""Create a career development plan based on:
            Position: {best_job['job_title']} at {best_job['company_name']}
            Location: {best_job['location']}
            Match Level: {best_match['match_level']}

            Technical Skills Gap: {', '.join(best_match['missing_skills']['technical'])}
            Professional Skills Gap: {', '.join(best_match['missing_skills']['professional'])}

            Minimum Qualifications: {best_job['minimum_qualifications']}

            Provide:
            1. Immediate steps to improve candidacy
            2. Long-term development recommendations
            3. Timeline for skill acquisition
            4. Additional certifications or qualifications to consider
            """
            development_response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "You are an expert career advisor."},
                    {"role": "user", "content": development_prompt}
                ]
            )
            development_plan = development_response.choices[0].message.content
            logging.info("Generated development plan")

            return {
                "job_match": best_match,
                "job_details": best_job.to_dict(),
                "course_recommendations": course_recommendations,
                "development_plan": development_plan
            }

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return None


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.getNumPages()):
            text += reader.getPage(page).extractText()
    return text


def main():
    try:
        # Initialize with your OpenAI API key
        api_key = ""
        advisor = CareerPathAdvisor(api_key)

        # Load necessary data
        jobs_df = pd.read_csv("ontario job.csv")
        courses_df = pd.read_csv("coursera_cleaned_courses1.csv")

        logging.info(f"Loaded {len(jobs_df)} Ontario jobs and {len(courses_df)} courses")

        resume_text = extract_text_from_pdf("resume.pdf")

        print("\n=== Starting Career Analysis ===")
        results = advisor.process_job_application(resume_text, jobs_df, courses_df)

        if results:
            print("\n=== Best Matching Job ===")
            print(f"Position: {results['job_details']['job_title']}")
            print(f"Company: {results['job_details']['company_name']}")
            print(f"Location: {results['job_details']['location']}")
            print(f"Job Type: {results['job_details']['job_type']}")
            print(f"Match Level: {results['job_match']['match_level']}")
            print(f"Reasoning: {results['job_match']['reasoning']}")

            print("\n=== Skills Analysis ===")
            print("Matching Technical Skills:",
                  ', '.join(results['job_match']['key_matching_skills']['technical']))
            print("Matching Professional Skills:",
                  ', '.join(results['job_match']['key_matching_skills']['professional']))
            print("\nTechnical Skills to Develop:",
                  ', '.join(results['job_match']['missing_skills']['technical']))
            print("Professional Skills to Develop:",
                  ', '.join(results['job_match']['missing_skills']['professional']))

            print("\n=== Qualification Match ===")
            print(f"Location Match: {results['job_match']['location_match']}")
            print(f"Qualification Match: {results['job_match']['qualification_match']}")

            if 'recommendations' in results['course_recommendations']:
                print("\n=== Recommended Courses ===")
                for course in results['course_recommendations']['recommendations']:
                    print(f"\nCourse: {course['course_name']}")
                    print(f"For Skill: {course['skill']}")
                    print(f"Relevance: {course['relevance_level']}")
                    print(f"Why This Course: {course['description']}")

            print("\n=== Career Development Plan ===")
            print(results['development_plan'])
        else:
            print("Error processing application. Please check the logs for details.")

    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")
        print("An error occurred. Please check the logs for details.")


if __name__ == "__main__":
    main()
