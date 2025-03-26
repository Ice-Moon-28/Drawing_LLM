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

svg_prompt_template_user = """Generate complete, well-formed SVG code that visually and artistically represents the following text description. The SVG should have a clear, balanced composition and high aesthetic quality. Please follow these constraints exactly:

<constraints>
* **Allowed Elements:** svg, path, circle, rect, ellipse, line, polyline, polygon, g, linearGradient, radialGradient, stop, defs
* **Allowed Attributes:** viewBox, width, height, fill, stroke, stroke-width, d, cx, cy, r, x, y, rx, ry, x1, y1, x2, y2, points, transform, opacity
</constraints>

<notes>
* Do not include any text elements or any lettering in the SVG.
* Focus solely on the visual representation of the description.
* Ensure that the generated SVG is aesthetically pleasing, with balanced proportions, harmonious colors, and clear visual structure.
* Provide the complete SVG code with no omissions or ellipses.
</notes>

<example>
<description>"A red circle with a blue square inside"</description>
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="128" cy="128" r="60" fill="red"/>
  <rect x="98" y="98" width="60" height="60" fill="blue"/>
</svg>
</example>
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

def generate_descrption_prompt(number):
    
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        { 
            "role": "user", 
            "content": """
            You're a creative assistant tasked with generating textual descriptions similar to the following examples.
            Examples:
            - a starlit night over snow-covered peaks
            - black and white checkered pants
            - crimson rectangles forming a chaotic grid
            - burgundy corduroy pants with patch pockets and silver buttons
            - orange corduroy overalls
            - a lighthouse overlooking the ocean
            - a green lagoon under a cloudy sky
            - a snowy plain
            - a maroon dodecahedron interwoven with teal threads
            - a purple silk scarf with tassel trim
            - magenta trapezoids layered on a translucent silver sheet
            - gray wool coat with a faux fur collar
            - a purple forest at dusk
            - purple pyramids spiraling around a bronze cone
            - khaki triangles and azure crescents

            Please generate {number} new, unique, and similarly styled descriptions. Wrap each description with <answer></answer> tags:
            """.format(number=number),
        } 
    ]