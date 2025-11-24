"""
Web Search Tool
Performs web searches to gather external context and information.
"""

import requests
from typing import Dict, Any, List, Optional
import logging
from bs4 import BeautifulSoup
import json

from .base_tool import BaseCustomTool
from ..utils.helpers import get_timestamp
from ..utils.validators import DataValidator
from pydantic import BaseModel, Field
from typing import Type


class WebSearchInput(BaseModel):
    """Input schema for Web Search Tool."""
    
    query: str = Field(description="Search query string")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchTool(BaseCustomTool):
    """
    Web search tool for gathering external information, context, and references
    to enrich data analysis with external knowledge.
    """
    
    name: str = "Web Search Tool"
    description: str = """Search the web for relevant information, context, and references 
    to support data analysis with external knowledge."""
    args_schema: Type[BaseModel] = WebSearchInput
    
    def execute(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute web search.
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: 5)
            **kwargs: Additional parameters (search_type, extract_content)
            
        Returns:
            Dict with search results and extracted information
        """
        if not query:
            raise ValueError("Search query is required")
        
        search_type = kwargs.get('search_type', 'general')
        extract_content = kwargs.get('extract_content', False)
        
        results = {
            'timestamp': get_timestamp(),
            'query': query,
            'search_type': search_type,
            'results': [],
            'summary': '',
            'insights': []
        }
        
        try:
            # Perform search using DuckDuckGo (no API key required)
            search_results = self._search_duckduckgo(query, num_results)
            
            for result in search_results:
                result_data = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('snippet', ''),
                    'source': result.get('source', 'web')
                }
                
                # Extract full content if requested
                if extract_content and result.get('url'):
                    content = self._extract_page_content(result['url'])
                    if content:
                        result_data['content'] = content
                        result_data['content_length'] = len(content)
                
                results['results'].append(result_data)
            
            # Generate summary
            results['summary'] = self._generate_search_summary(results)
            results['insights'] = self._extract_insights(results)
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'tool': self.name,
                    'total_results': len(results['results']),
                    'content_extracted': extract_content
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during web search: {str(e)}")
            # Fallback to mock results for demonstration
            return self._get_mock_results(query, num_results)
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Perform search using DuckDuckGo.
        This is a simplified implementation. In production, use proper API.
        """
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Abstract
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': data.get('AbstractSource', 'DuckDuckGo')
                })
            
            # Related topics
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo'
                    })
            
            return results[:num_results]
            
        except Exception as e:
            self.logger.warning(f"DuckDuckGo search failed: {str(e)}, using mock results")
            return []
    
    def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract main content from a web page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit length
            return text[:5000] if text else None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return None
    
    def _generate_search_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of search results."""
        num_results = len(results['results'])
        query = results['query']
        
        if num_results == 0:
            return f"No results found for query: '{query}'"
        
        summary_parts = [
            f"Found {num_results} results for '{query}'.",
        ]
        
        # Include first result snippet
        if results['results']:
            first_snippet = results['results'][0].get('snippet', '')[:200]
            if first_snippet:
                summary_parts.append(f"Top result: {first_snippet}...")
        
        return " ".join(summary_parts)
    
    def _extract_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from search results."""
        insights = []
        
        if not results['results']:
            return ["No external references found"]
        
        # Count unique sources
        sources = set(r.get('source', 'unknown') for r in results['results'])
        insights.append(f"Information gathered from {len(sources)} source(s)")
        
        # Check for content extraction
        content_extracted = sum(1 for r in results['results'] if 'content' in r)
        if content_extracted > 0:
            insights.append(f"Full content extracted from {content_extracted} page(s)")
        
        # Add context about result quality
        avg_snippet_length = sum(len(r.get('snippet', '')) for r in results['results']) / len(results['results'])
        if avg_snippet_length > 100:
            insights.append("Results contain detailed information")
        
        return insights
    
    def _get_mock_results(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Provide mock search results for demonstration purposes.
        Used when actual web search is unavailable.
        """
        mock_results = {
            'timestamp': get_timestamp(),
            'query': query,
            'search_type': 'mock',
            'results': [
                {
                    'title': f'Analysis of {query}: Comprehensive Guide',
                    'url': 'https://example.com/analysis-guide',
                    'snippet': f'This comprehensive guide explores various aspects of {query}, '
                              'including key trends, statistical patterns, and best practices for analysis.',
                    'source': 'Educational Resource'
                },
                {
                    'title': f'Understanding {query}: Key Insights and Trends',
                    'url': 'https://example.com/insights',
                    'snippet': f'Recent research on {query} reveals important patterns and correlations. '
                              'This article discusses methodologies and findings from recent studies.',
                    'source': 'Research Journal'
                },
                {
                    'title': f'{query}: Industry Best Practices',
                    'url': 'https://example.com/best-practices',
                    'snippet': f'Learn about industry standards and best practices for working with {query}. '
                              'Includes expert recommendations and case studies.',
                    'source': 'Industry Publication'
                }
            ][:num_results],
            'summary': f"Found {min(3, num_results)} mock results for demonstration purposes.",
            'insights': [
                'Mock results provided for demonstration',
                'In production, this would perform actual web searches',
                'Results include educational and research sources'
            ]
        }
        
        return {
            'success': True,
            'results': mock_results,
            'metadata': {
                'tool': self.name,
                'total_results': len(mock_results['results']),
                'note': 'Mock results for demonstration'
            }
        }
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate tool inputs."""
        query = kwargs.get('query')
        if not query or not isinstance(query, str):
            return False
        
        num_results = kwargs.get('num_results', 5)
        if not isinstance(num_results, int) or num_results < 1 or num_results > 20:
            return False
        
        search_type = kwargs.get('search_type', 'general')
        if search_type not in ['general', 'news', 'academic']:
            return False
        
        return True
