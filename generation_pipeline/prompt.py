svg_prompt_template = """You are a professional AI assistant skilled in generating SVG code. Your task is to generate an SVG image that strictly adheres to the following constraints:

1. **Allowed Elements & Attributes**: You must only use the following SVG elements and attributes:

   - Common attributes:
     {{'id', 'clip-path', 'clip-rule', 'color', 'color-interpolation', 'color-interpolation-filters', 
     'color-rendering', 'display', 'fill', 'fill-opacity', 'fill-rule', 'filter', 'flood-color', 'flood-opacity', 
     'lighting-color', 'marker-end', 'marker-mid', 'marker-start', 'mask', 'opacity', 'paint-order', 
     'stop-color', 'stop-opacity', 'stroke', 'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 
     'stroke-linejoin', 'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'transform'}}

   - SVG elements:
     {{'svg', 'g', 'defs', 'symbol', 'use', 'marker', 'pattern', 'linearGradient', 'radialGradient', 
     'stop', 'filter', 'feBlend', 'feFlood', 'feOffset', 'path', 'rect', 'circle', 'ellipse', 'line', 
     'polyline', 'polygon'}}

2. **Visual Theme**: Generate an SVG based on the following prompt:
   *"{description}"*
   
3. **Technical Constraints**:
   - Ensure the SVG uses proper `viewBox`, `width`, and `height` settings.
   - Use gradients, patterns, and filters where applicable, but only within the allowed elements and attributes.
   - The output should be **valid SVG code** that adheres to the constraints.

Generate the SVG **code only**, without explanations.
"""

def generate_svg_prompt(description):
    return svg_prompt_template.format(description=description)

if __name__ == "__main__":
    description = "A lighthouse overlooking the ocean"
    prompt = generate_svg_prompt(description)
    print(prompt)

description_prompt = """
You're a creative assistant tasked with generating textual descriptions similar to the following examples. Each description either depicts a natural scenery or describes clothing design elements with vivid colors and unique geometric patterns.

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

Please generate {number} new, unique, and similarly styled descriptions:
"""

def generate_descrption_prompt(number):
    return description_prompt.format(number=number)