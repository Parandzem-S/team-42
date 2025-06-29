import json
import os
from pathlib import Path

def main():
    print("🚀 Starting the notebook widget fix process...")
    
    # Check which notebook files exist
    notebook_files = list(Path('.').glob('*.ipynb'))
    print(f"📁 Found notebook files: {[f.name for f in notebook_files]}")
    
    # Try the original filename first
    input_file = 'ConvertDocumentsWithPhi4.ipynb'
    output_file = 'ConvertDocumentsWithPhi4.ipynb'
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"❌ File '{input_file}' not found!")
        

    
    try:
        # Create backup first
        backup_file = input_file.replace('.ipynb', '_backup.ipynb')
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📋 Created backup: {backup_file}")
        
        # Read the notebook
        with open(input_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print(f"📖 Successfully loaded notebook: {input_file}")
        
        # Check if widgets metadata exists
        widgets_removed = False
        if 'metadata' in notebook and 'widgets' in notebook['metadata']:
            widget_count = 0
            if 'application/vnd.jupyter.widget-state+json' in notebook['metadata']['widgets']:
                widget_state = notebook['metadata']['widgets']['application/vnd.jupyter.widget-state+json']
                if 'state' in widget_state:
                    widget_count = len(widget_state['state'])
            
            print(f"🔧 Found widgets metadata with {widget_count} widget objects")
            del notebook['metadata']['widgets']
            widgets_removed = True
        else:
            print("ℹ️  No widgets metadata found")
        
        # Also clean any cell outputs with widget data
        cells_cleaned = 0
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code' and 'outputs' in cell:
                original_outputs = len(cell['outputs'])
                # Remove outputs that contain widget view data
                cell['outputs'] = [
                    output for output in cell['outputs']
                    if not (output.get('data', {}).get('application/vnd.jupyter.widget-view+json'))
                ]
                if len(cell['outputs']) != original_outputs:
                    cells_cleaned += 1
        
        # Save the fixed notebook
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Notebook fixed and saved as: {output_file}")
        
        if widgets_removed:
            print("   - Removed problematic widgets metadata")
        if cells_cleaned > 0:
            print(f"   - Cleaned widget outputs from {cells_cleaned} cells")
        
        print("\n🎉 Your notebook should now render without widget errors!")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print("   The notebook file might be corrupted or not a valid JSON file")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Please check the file and try again")

if __name__ == "__main__":
    main()