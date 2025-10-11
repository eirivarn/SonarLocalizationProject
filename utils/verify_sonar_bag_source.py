#!/usr/bin/env python3
"""
Script to verify which bag files (data vs video) contain sonar data.

This script inspects actual bag files to confirm whether sonar topics are found
in *_data.bag or *_video.bag files, settling the question definitively.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re

# Import existing utilities
from utils.dataset_export_utils import find_data_bags, find_video_bags, list_topics_in_bag
from rosbags.highlevel import AnyReader


def find_sonar_topics_in_bag(bag_path: Path) -> List[Tuple[str, str]]:
    """
    Find all sonar-related topics in a bag file.
    
    Returns:
        List of (topic, msgtype) tuples for sonar-related topics
    """
    sonar_patterns = [
        r'/sonoptix', r'/echo', r'/sonar', r'/ping360', r'/ping\b', r'/mbes',
        r'sonoptix', r'echo', r'sonar', r'ping360', r'ping', r'mbes'
    ]
    sonar_regex = re.compile('|'.join(sonar_patterns), re.IGNORECASE)
    
    sonar_msg_types = {
        'SonoptixECHO', 'Ping360', 'Ping', 'Float32MultiArray'
    }
    
    try:
        topics = list_topics_in_bag(bag_path)
        sonar_topics = []
        
        for topic, msgtype in topics:
            # Check if topic name matches sonar patterns
            if sonar_regex.search(topic):
                sonar_topics.append((topic, msgtype))
            # Check if message type is sonar-related
            elif any(stype in msgtype for stype in sonar_msg_types):
                # Additional check - examine the topic name more carefully
                if any(pattern in topic.lower() for pattern in ['sensor', 'image', 'data']):
                    sonar_topics.append((topic, msgtype))
        
        return sonar_topics
    except Exception as e:
        print(f"   ❌ Error reading {bag_path.name}: {e}")
        return []


def get_topic_message_count(bag_path: Path, topic: str) -> int:
    """Get the number of messages for a specific topic in a bag."""
    try:
        with AnyReader([bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                return 0
            
            count = 0
            for _, _, _ in reader.messages(connections=connections):
                count += 1
            return count
    except Exception:
        return 0


def analyze_sonar_distribution(data_dir: Path, max_bags: int = 5) -> Dict:
    """
    Analyze sonar topic distribution across data and video bags.
    
    Args:
        data_dir: Directory containing bag files
        max_bags: Maximum number of bags to analyze (for performance)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Analyzing sonar data distribution in: {data_dir}")
    print("=" * 60)
    
    # Find bag files
    data_bags = find_data_bags(data_dir, recursive=True)
    video_bags = find_video_bags(data_dir, recursive=True)
    
    print(f"📁 Found {len(data_bags)} data bags and {len(video_bags)} video bags")
    
    if not data_bags and not video_bags:
        print("❌ No bag files found!")
        return {}
    
    # Limit analysis for performance
    data_bags_to_check = data_bags[:max_bags]
    video_bags_to_check = video_bags[:max_bags]
    
    results = {
        'data_bags_with_sonar': [],
        'video_bags_with_sonar': [],
        'all_sonar_topics': set(),
        'summary': {}
    }
    
    # Check data bags
    print(f"\n🔍 Checking data bags (analyzing {len(data_bags_to_check)} of {len(data_bags)}):")
    for bag in data_bags_to_check:
        print(f"   📦 {bag.name}")
        sonar_topics = find_sonar_topics_in_bag(bag)
        
        if sonar_topics:
            results['data_bags_with_sonar'].append(bag)
            print(f"      ✅ Found {len(sonar_topics)} sonar topics:")
            for topic, msgtype in sonar_topics:
                print(f"         - {topic} ({msgtype})")
                results['all_sonar_topics'].add(topic)
                
                # Get message count for this topic
                count = get_topic_message_count(bag, topic)
                if count > 0:
                    print(f"           📊 {count} messages")
        else:
            print(f"      ❌ No sonar topics found")
    
    # Check video bags  
    print(f"\n🔍 Checking video bags (analyzing {len(video_bags_to_check)} of {len(video_bags)}):")
    for bag in video_bags_to_check:
        print(f"   📦 {bag.name}")
        sonar_topics = find_sonar_topics_in_bag(bag)
        
        if sonar_topics:
            results['video_bags_with_sonar'].append(bag)
            print(f"      ✅ Found {len(sonar_topics)} sonar topics:")
            for topic, msgtype in sonar_topics:
                print(f"         - {topic} ({msgtype})")
                results['all_sonar_topics'].add(topic)
                
                # Get message count for this topic
                count = get_topic_message_count(bag, topic)
                if count > 0:
                    print(f"           📊 {count} messages")
        else:
            print(f"      ❌ No sonar topics found")
    
    # Generate summary
    data_sonar_count = len(results['data_bags_with_sonar'])
    video_sonar_count = len(results['video_bags_with_sonar'])
    
    print(f"\n📊 SUMMARY:")
    print("=" * 30)
    print(f"   📦 Data bags with sonar: {data_sonar_count}/{len(data_bags_to_check)}")
    print(f"   📦 Video bags with sonar: {video_sonar_count}/{len(video_bags_to_check)}")
    print(f"   🎯 Unique sonar topics found: {len(results['all_sonar_topics'])}")
    
    if results['all_sonar_topics']:
        print(f"   📋 All sonar topics:")
        for topic in sorted(results['all_sonar_topics']):
            print(f"      - {topic}")
    
    # Determine conclusion
    if video_sonar_count > 0 and data_sonar_count == 0:
        conclusion = "✅ CONFIRMED: Sonar data is in VIDEO bags only"
    elif data_sonar_count > 0 and video_sonar_count == 0:
        conclusion = "⚠️  UNEXPECTED: Sonar data is in DATA bags only"
    elif data_sonar_count > 0 and video_sonar_count > 0:
        conclusion = "🤔 MIXED: Sonar data found in BOTH data and video bags"
    else:
        conclusion = "❌ NO SONAR DATA FOUND in analyzed bags"
    
    print(f"\n🎯 CONCLUSION: {conclusion}")
    
    results['summary'] = {
        'data_bags_analyzed': len(data_bags_to_check),
        'video_bags_analyzed': len(video_bags_to_check),
        'data_bags_with_sonar': data_sonar_count,
        'video_bags_with_sonar': video_sonar_count,
        'total_unique_topics': len(results['all_sonar_topics']),
        'conclusion': conclusion
    }
    
    return results


def main():
    """Main function to run the verification."""
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Use current directory or default data directory
        data_dir = Path.cwd()
        
        # Check if we're in the SOLAQUA directory, use a reasonable default
        if (data_dir / "utils").exists() and (data_dir / "README.md").exists():
            # We're likely in the SOLAQUA project root
            # You might want to adjust this path to where your actual bag files are
            possible_data_dirs = [
                Path("/Volumes/LaCie/SOLAQUA/data"),  # External drive
                data_dir / "data",                     # Local data folder
                data_dir.parent / "data",             # Parent data folder
            ]
            
            for possible_dir in possible_data_dirs:
                if possible_dir.exists():
                    data_dir = possible_dir
                    break
            else:
                print("❌ No bag data directory found. Please specify path as argument:")
                print("   python verify_sonar_bag_source.py /path/to/bag/files")
                return
    
    if not data_dir.exists():
        print(f"❌ Directory does not exist: {data_dir}")
        print("Usage: python verify_sonar_bag_source.py [/path/to/bag/files]")
        return
    
    print(f"🚀 SOLAQUA Sonar Data Source Verification")
    print(f"📁 Analyzing bags in: {data_dir}")
    
    # Run the analysis
    results = analyze_sonar_distribution(data_dir, max_bags=10)
    
    # Optional: Save results to file
    if results:
        import json
        results_file = data_dir / "sonar_verification_results.json"
        
        # Convert Path objects to strings for JSON serialization
        json_results = results.copy()
        json_results['data_bags_with_sonar'] = [str(p) for p in results['data_bags_with_sonar']]
        json_results['video_bags_with_sonar'] = [str(p) for p in results['video_bags_with_sonar']]
        json_results['all_sonar_topics'] = list(results['all_sonar_topics'])
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()