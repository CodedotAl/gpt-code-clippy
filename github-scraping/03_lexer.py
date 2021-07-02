import os
import sys
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

def main():
    if len(sys.argv) <= 3:
        raise ValueError('Provide at least an input directory, an output directory, and a language.')
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    language = sys.argv[3]
    if input_directory.endswith('/'):
        input_directory = input_directory[:-1]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    lexer = get_lexer_by_name(language)
    language_extensions = set(ext.lower()[1:] for ext in lexer.filenames)

    with open(os.path.join(output_directory, os.path.basename(input_directory) + '.txt'), 'w') as f_out:
        for root, _, files in os.walk(input_directory):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in language_extensions:
                    print(f'Lexing: {root}, {name}')
                    lex_file(os.path.join(root, name), f_out, lexer)


def lex_file(file_path, f_out, lexer):
    with open(file_path, errors='ignore') as f_in:
        text = f_in.read()

        lexed = []
        for (ttype, token) in pygments.lex(text, lexer):
            if ttype in Token.Text:
                continue
            elif ttype in Token.Comment:
                continue
            else:
                lexed.append(token.replace('\t', '#TAB#'))

        # Skip empty files.
        if not lexed:
            return
        f_out.write('\t'.join(lexed))
        f_out.write('\n')

if __name__ == '__main__':
    main()