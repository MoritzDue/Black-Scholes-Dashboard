from tkinter import Tk, Label, Scale, Button, HORIZONTAL, Text, END, Toplevel, font
import numpy as np
from scipy.stats import norm

def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

def on_calculate():
    premium = slider_premium.get()
    S = slider_S.get()
    K = slider_K.get()
    T = slider_T.get()
    r = slider_r.get() / 100  # Convert percent to decimal

    # Create new window for results
    result_window = Toplevel(root)
    result_window.title("Black-Scholes Results")
    result_window.configure(bg="#f0f8ff")
    # Show initial inputs with colored labels
    header_font = font.Font(family="Helvetica", size=10, weight="bold")
    Label(result_window, text=f"Premium: {premium}", bg="#e6f7ff", fg="#007acc", font=header_font, padx=10, pady=5).grid(row=0, column=0, sticky="ew")
    Label(result_window, text=f"Underlying Price (S): {S}", bg="#e6f7ff", fg="#007acc", font=header_font, padx=10, pady=5).grid(row=0, column=1, sticky="ew")
    Label(result_window, text=f"Strike Price (K): {K}", bg="#e6f7ff", fg="#007acc", font=header_font, padx=10, pady=5).grid(row=0, column=2, sticky="ew")
    Label(result_window, text=f"Time to Expiration (T in years): {T}", bg="#e6f7ff", fg="#007acc", font=header_font, padx=10, pady=5).grid(row=1, column=0, sticky="ew")
    Label(result_window, text=f"Risk-Free Rate (r): {r:.2%}", bg="#e6f7ff", fg="#007acc", font=header_font, padx=10, pady=5).grid(row=1, column=1, sticky="ew")

    # Table for results
    table = Text(result_window, width=45, height=13, bg="#f8f8ff", fg="#333366", font=("Consolas", 10))
    table.grid(row=2, columnspan=3, padx=10, pady=10)
    table.insert(END, f"{'Volatility':<12}{'Call P&L':<15}{'Put P&L':<15}\n")
    table.insert(END, "="*42 + "\n")
    for i in range(1, 11):  # Start at 0.10
        vol = i * 0.10
        C, P = calculate_option_prices(S, K, T, r, vol, premium)
        table.insert(END, f"{vol:<12.2f}{C:<15.2f}{P:<15.2f}\n")

root = Tk()
root.title("Black-Scholes Option Pricing")
root.configure(bg="#e6f7ff")

slider_font = font.Font(family="Helvetica", size=10, weight="bold")

Label(root, text="Premium:", bg="#e6f7ff", fg="#007acc", font=slider_font, padx=10, pady=5).grid(row=0, column=0, sticky="ew")
slider_premium = Scale(root, from_=0, to=5, orient=HORIZONTAL, resolution=0.01, length=250, bg="#e6f7ff", fg="#007acc")
slider_premium.set(2.5)
slider_premium.grid(row=0, column=1, padx=5, pady=5)

Label(root, text="Underlying Price (S):", bg="#e6f7ff", fg="#007acc", font=slider_font, padx=10, pady=5).grid(row=1, column=0, sticky="ew")
slider_S = Scale(root, from_=0, to=1000, orient=HORIZONTAL, resolution=1, length=250, bg="#e6f7ff", fg="#007acc")
slider_S.set(50)
slider_S.grid(row=1, column=1, padx=5, pady=5)

Label(root, text="Strike Price (K):", bg="#e6f7ff", fg="#007acc", font=slider_font, padx=10, pady=5).grid(row=2, column=0, sticky="ew")
slider_K = Scale(root, from_=0, to=1000, orient=HORIZONTAL, resolution=1, length=250, bg="#e6f7ff", fg="#007acc")
slider_K.set(55)
slider_K.grid(row=2, column=1, padx=5, pady=5)

Label(root, text="Time to Expiration (T in years):", bg="#e6f7ff", fg="#007acc", font=slider_font, padx=10, pady=5).grid(row=3, column=0, sticky="ew")
slider_T = Scale(root, from_=0, to=10, resolution=0.01, orient=HORIZONTAL, length=250, bg="#e6f7ff", fg="#007acc")
slider_T.set(1)
slider_T.grid(row=3, column=1, padx=5, pady=5)

Label(root, text="Risk-Free Rate (r) [%]:", bg="#e6f7ff", fg="#007acc", font=slider_font, padx=10, pady=5).grid(row=4, column=0, sticky="ew")
slider_r = Scale(root, from_=0, to=100, resolution=0.01, orient=HORIZONTAL, length=250, bg="#e6f7ff", fg="#007acc")
slider_r.set(2)
slider_r.grid(row=4, column=1, padx=5, pady=5)

Button(root, text="Calculate", command=on_calculate, bg="#007acc", fg="white", font=slider_font, padx=10, pady=5).grid(row=5, columnspan=2, pady=10)

root.mainloop()