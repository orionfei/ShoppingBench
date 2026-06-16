import pdfplumber

pdf = pdfplumber.open('ShoppingBench.pdf')
print(f'Total pages: {len(pdf.pages)}')

# 读取所有页面
text = ''
for i in range(len(pdf.pages)):
    text += f'\n\n=== PAGE {i+1} ===\n\n'
    text += pdf.pages[i].extract_text()

# 保存到文件
with open('paper_content.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Paper content saved to paper_content.txt")
print(f"Total characters: {len(text)}")

# 打印前10000个字符
print("\n" + "="*50)
print("FIRST 10000 CHARACTERS:")
print("="*50)
print(text[:10000])