import pdfplumber

pdf = pdfplumber.open('ShoppingBench.pdf')
print(f'Total pages: {len(pdf.pages)}')

text = ''
pages_to_read = min(10, len(pdf.pages))
for i in range(pages_to_read):
    text += pdf.pages[i].extract_text() + '\n\n'

print(text[:5000])