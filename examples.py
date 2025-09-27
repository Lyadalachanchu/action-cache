#!/usr/bin/env python3
"""
Example usage of Playwright with Lightpanda remote browser
This script demonstrates various browser automation tasks using Lightpanda's CDP service
"""

import os
import time
from playwright.sync_api import sync_playwright
from main import LightpandaCDPClient

def example_basic_navigation():
    """Basic navigation and screenshot example"""
    print("=== Basic Navigation Example ===")
    
    cdp_client = LightpandaCDPClient()
    
    with sync_playwright() as p:
        browser = cdp_client.connect_browser(p)
        page = browser.new_page()
        
        # Navigate to a website
        page.goto('https://example.com')
        print(f"Page title: {page.title()}")
        
        # Take screenshot
        page.screenshot(path='example-basic.png')
        print("Screenshot saved as 'example-basic.png'")
        
        browser.close()

def example_form_interaction():
    """Example of form interaction and data extraction"""
    print("\n=== Form Interaction Example ===")
    
    cdp_client = LightpandaCDPClient()
    
    with sync_playwright() as p:
        browser = cdp_client.connect_browser(p)
        page = browser.new_page()
        
        # Navigate to a form page (using httpbin for testing)
        page.goto('https://httpbin.org/forms/post')
        
        # Fill out form fields
        page.fill('input[name="custname"]', 'John Doe')
        page.fill('input[name="custtel"]', '555-1234')
        page.fill('input[name="custemail"]', 'john@example.com')
        page.select_option('select[name="size"]', 'large')
        
        # Take screenshot before submission
        page.screenshot(path='form-filled.png')
        print("Form filled and screenshot saved as 'form-filled.png'")
        
        browser.close()

def example_async_operations():
    """Example of handling async operations and waiting"""
    print("\n=== Async Operations Example ===")
    
    cdp_client = LightpandaCDPClient()
    
    with sync_playwright() as p:
        browser = cdp_client.connect_browser(p)
        page = browser.new_page()
        
        # Navigate to a page with dynamic content
        page.goto('https://playwright.dev')
        
        # Wait for specific element to load
        page.wait_for_selector('h1')
        
        # Get all links on the page
        links = page.query_selector_all('a')
        print(f"Found {len(links)} links on the page")
        
        # Extract text content
        heading = page.query_selector('h1')
        if heading:
            print(f"Main heading: {heading.text_content()}")
        
        page.screenshot(path='async-example.png')
        print("Screenshot saved as 'async-example.png'")
        
        browser.close()

def example_multiple_pages():
    """Example of working with multiple pages/tabs"""
    print("\n=== Multiple Pages Example ===")
    
    cdp_client = LightpandaCDPClient()
    
    with sync_playwright() as p:
        browser = cdp_client.connect_browser(p)
        
        # Create multiple pages
        page1 = browser.new_page()
        page2 = browser.new_page()
        
        # Navigate to different sites
        page1.goto('https://github.com')
        page2.goto('https://stackoverflow.com')
        
        # Take screenshots of both pages
        page1.screenshot(path='github-page.png')
        page2.screenshot(path='stackoverflow-page.png')
        
        print("Screenshots saved for both pages")
        
        browser.close()

def main():
    """Run all examples"""
    print("Starting Lightpanda Playwright Examples")
    print("Make sure your LIGHTPANDA_TOKEN is set in your environment")
    
    try:
        # Run examples
        example_basic_navigation()
        example_form_interaction()
        example_async_operations()
        example_multiple_pages()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Please check your Lightpanda token and connection")

if __name__ == "__main__":
    main()
