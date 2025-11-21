"""
HTML report generator for photo analysis results.
Creates visual reports showing input images, outputs, and analysis results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64


def _encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 data URI."""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            ext = image_path.suffix.lower()
            mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else f'image/{ext[1:]}'
            return f"data:{mime_type};base64,{base64.b64encode(img_data).decode('utf-8')}"
    except Exception:
        return ""


def _group_tags_by_category(analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tags by category for organized display."""
    tags = analysis.get('tags', {})
    category_scores = analysis.get('category_scores', {})
    
    # Category definitions (matching prompts config)
    categories = {
        'quality_technical': {
            'name': 'Photo Quality',
            'description': 'Technical quality indicators',
            'prompts': [
                "a high-quality photo", "a low-quality photo", "a blurry photo",
                "a sharp photo", "a noisy image", "a clean image",
                "a well-exposed photo", "an underexposed photo", "an overexposed photo",
                "a photo with good lighting", "a photo with bad lighting", "a color photo"
            ]
        },
        'composition_aesthetic': {
            'name': 'Composition',
            'description': 'Aesthetic and composition features',
            'prompts': [
                "a well-composed photograph", "a poorly composed photograph",
                "a balanced composition", "an unbalanced composition",
                "a centered composition", "a rule of thirds composition",
                "a minimalist composition", "a cluttered composition",
                "a photo with good framing", "a photo with bad framing",
                "a photo with leading lines", "a photo with symmetry",
                "a photo with depth and perspective", "a flat composition"
            ]
        },
        'scene_content': {
            'name': 'Scene Content',
            'description': 'What\'s in the image',
            'prompts': [
                "a photo of a person", "a group photo", "a close-up face portrait",
                "a crowd of people", "a photo of a landmark", "a city skyline",
                "an indoor room", "an outdoor landscape", "a house exterior",
                "a mountain landscape", "a beach scene", "a photo of food",
                "a document or text page", "a drawing or painting", "a product photo",
                "a pet or animal", "a vehicle", "a building interior",
                "a night scene", "a daytime scene"
            ]
        },
        'human_focused': {
            'name': 'Human-Focused',
            'description': 'People and faces (triggers face detection, quality assessment)',
            'prompts': [
                "a selfie", "a portrait of one person", "a portrait of two people",
                "a group of people", "a person standing", "a person sitting",
                "a person smiling", "a person in profile", "a photo with multiple faces",
                "a person looking away from camera", "a person with eyes shut"
            ]
        }
    }
    
    grouped = {}
    
    for category_key, category_info in categories.items():
        category_tags = []
        for prompt in category_info['prompts']:
            if prompt in tags:
                category_tags.append({
                    'tag': prompt,
                    'score': tags[prompt]
                })
        
        # Sort by score and take top 5
        category_tags.sort(key=lambda x: x['score'], reverse=True)
        category_tags = category_tags[:5]
        
        if category_tags:
            grouped[category_key] = {
                'name': category_info['name'],
                'description': category_info['description'],
                'tags': [{'tag': t['tag'], 'score': f'{t["score"]:.3f}'} for t in category_tags],
                'category_score': f'{category_scores.get(category_key, 0.0):.3f}'
            }
    
    return grouped


def _get_routing_tags(analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Get tags that triggered each routing decision."""
    routing_tags = {}
    tags = analysis.get('tags', {})
    routing = analysis.get('routing', {})
    
    # Face detection tags
    if routing.get('needs_face_detection', False):
        face_tags = []
        if tags.get("a photo of a person", 0) > 0.6:
            face_tags.append({'tag': "a photo of a person", 'score': tags["a photo of a person"]})
        human_focused = [
            "a selfie", "a portrait of one person", "a portrait of two people",
            "a group of people", "a person standing", "a person sitting",
            "a person smiling", "a person in profile", "a photo with multiple faces"
        ]
        for tag in human_focused:
            if tags.get(tag, 0) > 0.6:
                face_tags.append({'tag': tag, 'score': tags[tag]})
        if face_tags:
            routing_tags['Face Detection'] = face_tags
    
    # Landmark detection tags
    if routing.get('needs_landmark_detection', False):
        landmark_score = tags.get("a photo of a landmark", 0)
        if landmark_score > 0.6:
            routing_tags['Landmark Detection'] = [
                {'tag': "a photo of a landmark", 'score': landmark_score}
            ]
    
    # Object detection tags
    if routing.get('needs_object_detection', False):
        object_tags = []
        if tags.get("a product photo", 0) > 0.5:
            object_tags.append({'tag': "a product photo", 'score': tags["a product photo"]})
        if tags.get("a vehicle", 0) > 0.5:
            object_tags.append({'tag': "a vehicle", 'score': tags["a vehicle"]})
        if object_tags:
            routing_tags['Object Detection'] = object_tags
    
    # Text detection tags
    if routing.get('needs_text_detection', False):
        text_score = tags.get("a document or text page", 0)
        if text_score > 0.7:
            routing_tags['Text Detection'] = [
                {'tag': "a document or text page", 'score': text_score}
            ]
    
    return routing_tags


def _format_routing(routing: Dict[str, bool]) -> List[str]:
    """Format routing decisions for display."""
    return [key.replace('needs_', '').replace('_', ' ').title() 
            for key, value in routing.items() if value]


def generate_html_report(
    analysis_results: Dict[str, Dict[str, Any]],
    original_images: Optional[Dict[str, Path]] = None,
    output_path: Path = Path("outputs/demo_report.html"),
    title: str = "Photo Analysis Report"
) -> Path:
    """
    Generate HTML report from analysis results.
    
    Args:
        analysis_results: Dictionary mapping image paths to analysis results
        original_images: Optional mapping of analysis paths to original image paths
        output_path: Path to save HTML report
        title: Report title
        
    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare image data
    image_cards = []
    total_images = len(analysis_results)
    successful = 0
    failed = 0
    
    for analysis_path, analysis in analysis_results.items():
        try:
            analysis_path_obj = Path(analysis_path)
            original_path = original_images.get(analysis_path, analysis_path_obj) if original_images else analysis_path_obj
            
            # Check if images exist
            has_original = original_path.exists()
            has_analysis = analysis_path_obj.exists()
            
            if not has_original and not has_analysis:
                failed += 1
                continue
                
            successful += 1
            
            # Get image data
            original_img = _encode_image_to_base64(original_path) if has_original else ""
            analysis_img = _encode_image_to_base64(analysis_path_obj) if has_analysis else ""
            
            # Format analysis data
            importance_score = analysis.get('importance_score', 0.0)
            primary_tags = analysis.get('primary_tags', [])
            tags_by_category = _group_tags_by_category(analysis)
            routing = _format_routing(analysis.get('routing', {}))
            routing_tags = _get_routing_tags(analysis)
            
            # Determine status
            status = "success" if importance_score > 0 else "warning"
            
            image_cards.append({
                'original_path': str(original_path),
                'analysis_path': str(analysis_path),
                'original_img': original_img,
                'analysis_img': analysis_img,
                'has_original': has_original,
                'has_analysis': has_analysis,
                'importance_score': f'{importance_score:.3f}',
                'primary_tags': primary_tags[:3],
                'tags_by_category': tags_by_category,
                'routing': routing,
                'routing_tags': routing_tags,
                'status': status
            })
        except Exception as e:
            failed += 1
            image_cards.append({
                'original_path': str(analysis_path),
                'analysis_path': str(analysis_path),
                'error': str(e),
                'status': 'error'
            })
    
    # Compute statistics
    importance_scores = [
        float(analysis.get('importance_score', 0))
        for analysis in analysis_results.values()
    ]
    
    stats = {
        'total': total_images,
        'successful': successful,
        'failed': failed,
        'min_importance': f'{min(importance_scores):.3f}' if importance_scores else '0.000',
        'max_importance': f'{max(importance_scores):.3f}' if importance_scores else '0.000',
        'avg_importance': f'{sum(importance_scores) / len(importance_scores):.3f}' if importance_scores else '0.000'
    }
    
    # Generate HTML
    html_content = _generate_html_content(title, stats, image_cards)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def _generate_html_content(title: str, stats: Dict[str, str], image_cards: List[Dict[str, Any]]) -> str:
    """Generate HTML content."""
    
    status_badges = {
        'success': '<span class="badge success">✓ Success</span>',
        'warning': '<span class="badge warning">⚠ Warning</span>',
        'error': '<span class="badge error">✗ Error</span>'
    }
    
    cards_html = []
    for card in image_cards:
        if card.get('error'):
            cards_html.append(f"""
            <div class="card error">
                <div class="card-header">
                    <h3>{Path(card['original_path']).name}</h3>
                    {status_badges.get(card['status'], '')}
                </div>
                <div class="card-body">
                    <p class="error-message">Error: {card['error']}</p>
                </div>
            </div>
            """)
        else:
            routing_html = ''.join([f'<span class="route-tag">{r}</span>' for r in card['routing']])
            
            # Build tags by category HTML
            tags_by_category_html = ""
            if card.get('tags_by_category'):
                category_order = ['quality_technical', 'composition_aesthetic', 'scene_content', 'human_focused']
                for cat_key in category_order:
                    if cat_key in card['tags_by_category']:
                        cat_data = card['tags_by_category'][cat_key]
                        cat_tags_html = ''.join([
                            f'<div class="tag-item"><span class="tag-name">{t["tag"]}</span><span class="tag-score">{t["score"]}</span></div>'
                            for t in cat_data['tags']
                        ])
                        tags_by_category_html += f'''
                            <div class="category-section">
                                <div class="category-header">
                                    <strong>{cat_data["name"]}</strong>
                                    <span class="category-score">Score: {cat_data["category_score"]}</span>
                                </div>
                                <div class="category-description">{cat_data["description"]}</div>
                                <div class="tags-list">{cat_tags_html}</div>
                            </div>
                        '''
            
            # Build routing tags HTML
            routing_tags_html = ""
            if card.get('routing_tags'):
                for model_name, trigger_tags in card['routing_tags'].items():
                    trigger_tags_html = ''.join([
                        f'<div class="routing-tag-item"><span class="routing-tag-name">{t["tag"]}</span><span class="routing-tag-score">{float(t["score"]):.3f}</span></div>'
                        for t in trigger_tags
                    ])
                    routing_tags_html += f'''
                        <div class="routing-model">
                            <strong>{model_name}:</strong>
                            <div class="routing-tags-list">{trigger_tags_html}</div>
                        </div>
                    '''
            
            images_html = ""
            if card['has_original']:
                images_html += f'<div class="image-container"><h4>Original</h4><img src="{card["original_img"]}" alt="Original"></div>'
            if card['has_analysis']:
                images_html += f'<div class="image-container"><h4>Thumbnail</h4><img src="{card["analysis_img"]}" alt="Analysis"></div>'
            
            cards_html.append(f"""
            <div class="card {card['status']}">
                <div class="card-header">
                    <h3>{Path(card['original_path']).name}</h3>
                    {status_badges.get(card['status'], '')}
                </div>
                <div class="card-body">
                    <div class="images-row">
                        {images_html}
                    </div>
                    <div class="info-section">
                        <div class="info-item">
                            <strong>Importance Score:</strong> <span class="score">{card['importance_score']}</span>
                        </div>
                        {f'<div class="info-item"><strong>Routing Models:</strong> <div class="routing">{routing_html}</div></div>' if card['routing'] else ''}
                        {f'<div class="info-item tags-by-category">{tags_by_category_html}</div>' if tags_by_category_html else ''}
                        {f'<div class="info-item routing-tags-section"><strong>Tags That Triggered Routing:</strong>{routing_tags_html}</div>' if routing_tags_html else ''}
                    </div>
                </div>
            </div>
            """)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        .card.success {{
            border-left: 4px solid #27ae60;
        }}
        
        .card.warning {{
            border-left: 4px solid #f39c12;
        }}
        
        .card.error {{
            border-left: 4px solid #e74c3c;
        }}
        
        .card-header {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .card-header h3 {{
            font-size: 16px;
            color: #2c3e50;
            margin: 0;
        }}
        
        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge.error {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .card-body {{
            padding: 20px;
        }}
        
        .images-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .image-container {{
            flex: 1;
        }}
        
        .image-container h4 {{
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 8px;
        }}
        
        .image-container img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        
        .info-section {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .info-item {{
            font-size: 14px;
        }}
        
        .info-item strong {{
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }}
        
        .score {{
            font-size: 18px;
            font-weight: bold;
            color: #3498db;
        }}
        
        .tags-by-category {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .category-section {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        
        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .category-header strong {{
            color: #2c3e50;
            font-size: 14px;
        }}
        
        .category-score {{
            font-size: 12px;
            color: #7f8c8d;
            font-weight: 600;
        }}
        
        .category-description {{
            font-size: 11px;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        .tags-list {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .tag-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            background: white;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .tag-name {{
            color: #2c3e50;
        }}
        
        .tag-score {{
            color: #7f8c8d;
            font-weight: 600;
        }}
        
        .routing {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 5px;
        }}
        
        .route-tag {{
            padding: 4px 10px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }}
        
        .routing-tags-section {{
            margin-top: 15px;
        }}
        
        .routing-model {{
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}
        
        .routing-model strong {{
            color: #2c3e50;
            display: block;
            margin-bottom: 8px;
        }}
        
        .routing-tags-list {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .routing-tag-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 8px;
            background: white;
            border-radius: 3px;
            font-size: 11px;
        }}
        
        .routing-tag-name {{
            color: #2c3e50;
            font-style: italic;
        }}
        
        .routing-tag-score {{
            color: #27ae60;
            font-weight: 600;
        }}
        
        .error-message {{
            color: #e74c3c;
            padding: 10px;
            background: #f8d7da;
            border-radius: 4px;
            font-size: 13px;
        }}
        
        @media (max-width: 768px) {{
            .cards-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{stats['total']}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #27ae60;">{stats['successful']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #e74c3c;">{stats['failed']}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['avg_importance']}</div>
                    <div class="stat-label">Avg Importance</div>
                </div>
            </div>
        </div>
        
        <div class="cards-grid">
            {''.join(cards_html)}
        </div>
    </div>
</body>
</html>
"""

