import sys,re

MAX_TEXT_INLINE = 250

def trim_text(text):
    """
    Trim text
    @param text:
    @return:
    """
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n+', '\n', text)

    return text


def limit_line_length(text):
    """
    Limit line length
    @param text:
    @return:
    """
    lines = []
    for line in text.split('\n'):
        chunks = [line[i:i+MAX_TEXT_INLINE]+'\n' for i in range(0, len(line), MAX_TEXT_INLINE)]
        lines.extend(chunks)
    return '\n'.join(lines)

f = open(sys.argv[1],"r")
lines = f.readlines()
for line in lines:
    if len(line)>3:
        print(limit_line_length(trim_text(line)))
