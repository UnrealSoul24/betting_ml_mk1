
try:
    content = ""
    try:
        with open('backtest_verification.log', 'r', encoding='utf-16') as f:
            content = f.read()
    except UnicodeError:
        try:
            with open('backtest_verification.log', 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeError:
            with open('backtest_verification.log', 'r', encoding='mbcs') as f:
                content = f.read()

    with open('backtest_final.txt', 'w', encoding='utf-8') as f:
        f.write(content)
        
except FileNotFoundError:
    print("Log file not found.")
