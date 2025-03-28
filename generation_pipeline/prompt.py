svg_prompt_template_system = """You are a professional AI assistant skilled in generating SVG code. Your task is to generate an SVG image that strictly adheres to the following constraints:"""

# svg_prompt_template_user = """
# 1. **Allowed Elements & Attributes**: You must only use the following SVG elements and attributes:

#    - Common attributes:
#      {{'id', 'clip-path', 'clip-rule', 'color', 'color-interpolation', 'color-interpolation-filters', 
#      'color-rendering', 'display', 'fill', 'fill-opacity', 'fill-rule', 'filter', 'flood-color', 'flood-opacity', 
#      'lighting-color', 'marker-end', 'marker-mid', 'marker-start', 'mask', 'opacity', 'paint-order', 
#      'stop-color', 'stop-opacity', 'stroke', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 
#      'stroke-linejoin', 'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'transform'}}

#    - SVG elements:
#      {{'svg', 'g', 'defs', 'symbol', 'use', 'marker', 'pattern', 'linearGradient', 'radialGradient', 
#      'stop', 'filter', 'feBlend', 'feFlood', 'feOffset', 'path', 'rect', 'circle', 'ellipse', 'line', 
#      'polyline', 'polygon'}}

# 2. **Visual Theme**: Generate an SVG based on the following prompt:
#    *"{description}"*
   
# 3. **Technical Constraints**:
#    - Ensure the SVG uses proper `viewBox`, `width`, and `height` settings.
#    - Use gradients, patterns, and filters where applicable, but only within the allowed elements and attributes.
#    - The output should be **valid SVG code** that adheres to the constraints.

# Generate the SVG **code only**, without explanations.
# """

svg_prompt_template_user = """You are to generate complete and well-formed SVG code that **artistically and accurately visualizes** the following description. Your output must strictly follow the format and constraints below.

<description>"{description}"</description>

<requirements>
1. **Visual Quality**: The SVG should be aesthetically pleasing, with:
   - Balanced composition
   - Harmonious and thoughtful use of colors
   - Clear, intentional structure

2. **Allowed Elements** (only these are permitted):
   svg, path, circle, rect, ellipse, line, polyline, polygon, g, linearGradient, radialGradient, stop, defs

3. **Allowed Attributes** (only these are permitted):
   viewBox, width, height, fill, stroke, stroke-width, d, cx, cy, r, x, y, rx, ry, x1, y1, x2, y2, points, transform, opacity

4. **Do Not Use**:
   - Any <text> or text-related elements
   - Any external assets or images
   - Any scripting or animation (e.g., no <script> or <animate>)

5. **Output Format**:
   - Provide a complete, standalone SVG code block
   - No ellipses, comments, or placeholders
   - Only return the SVG code (no additional explanation)

<example>
<description>"A red circle with a blue square inside"</description>
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="128" cy="128" r="60" fill="red"/>
  <rect x="98" y="98" width="60" height="60" fill="blue"/>
</svg>
</example>

Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.
Your SVG should be a faithful and creative interpretation of the description above, using only the allowed elements and attributes.
<description>"{description}"</description>
"""
# svg_prompt_template_user = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
# <constraints>
# * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
# * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
# </constraints>

# <example>
# <description>"A red circle with a blue square inside"</description>
# ```svg
# <svg viewBox="0 0 256 256" width="256" height="256">
#   <circle cx="50" cy="50" r="40" fill="red"/>
#   <rect x="30" y="30" width="40" height="40" fill="blue"/>
# </svg>
# ```
# </example>


# Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

# <description>"{description}"</description>
# ```svg
# <svg viewBox="0 0 256 256" width="256" height="256">
# """

def generate_svg_prompt(description):
    
    return [
        {"role": "system", "content": svg_prompt_template_system},
        {"role": "user", "content": svg_prompt_template_user.format(description=description)},
    ]

if __name__ == "__main__":
    description = "A lighthouse overlooking the ocean"
    prompt = generate_svg_prompt(description)
    print(prompt)

# description_prompt_system = """"""

def generate_description_prompt(number):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""
You're a creative assistant tasked with generating short, vivid, and imaginative textual descriptions. Each description must depict a **scene where a distinct object is set within a specific landscape or atmospheric environment**.

The object can be abstract (like a geometric shape, sculpture, or material form), or natural (like a tree, rock, or artifact). The setting should be scenic, atmospheric, or natural (like forests, oceans, skies, deserts, etc).

Examples:
- a maroon dodecahedron interwoven with teal threads on a snowy plain  
- a green lagoon under a cloudy sky  
- purple pyramids spiraling around a bronze cone  
- a lighthouse overlooking the ocean  
- crimson rectangles forming a chaotic grid over misty hills  
- an obsidian obelisk standing alone in a golden field at dusk  
- magenta trapezoids layered on a translucent silver sheet in the moonlight  
- a silver sphere suspended over a crimson desert  

Please generate {number} new, unique, and similarly styled descriptions that combine a **visually distinctive object** with a **natural or atmospheric setting**. Wrap each description with <answer></answer> tags.
"""
        }
    ]