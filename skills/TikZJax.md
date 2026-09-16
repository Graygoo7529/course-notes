<img width=275 align="right" src="./imgs/screenshot.png">

# Obsidian TikZJax

A plugin for Obsidian that lets you render LaTeX and TikZ diagrams in your notes.

You can render graphs, figures, circuits, chemical diagrams, commutative diagrams, and more.

The following packages are available in `\usepackage{}`:
- chemfig
- tikz-cd
- circuitikz
- pgfplots
- array
- amsmath
  - amstext
- amsfonts
- amssymb
- tikz-3dplot

## Usage
Content inside of `tikz` code blocks will be rendered by TikZJax.

- Remember to load any packages you need with `\usepackage{}`, and include `\begin{document}` and `\end{document}`.

- The standalone document class is used (`\documentclass{standalone}`).


### Examples
<img width=300 align="right" src="./imgs/img1.png">

````latex
```tikz
\begin{document}
  \begin{tikzpicture}[domain=0:4]
    \draw[very thin,color=gray] (-0.1,-1.1) grid (3.9,3.9);
    \draw[->] (-0.2,0) -- (4.2,0) node[right] {$x$};
    \draw[->] (0,-1.2) -- (0,4.2) node[above] {$f(x)$};
    \draw[color=red]    plot (\x,\x)             node[right] {$f(x) =x$};
    \draw[color=blue]   plot (\x,{sin(\x r)})    node[right] {$f(x) = \sin x$};
    \draw[color=orange] plot (\x,{0.05*exp(\x)}) node[right] {$f(x) = \frac{1}{20} \mathrm e^x$};
  \end{tikzpicture}
\end{document}
```
````

<img width=325 align="right" src="./imgs/img2.png">

````latex
```tikz
\usepackage{circuitikz}
\begin{document}

\begin{circuitikz}[american, voltage shift=0.5]
\draw (0,0)
to[isource, l=$I_0$, v=$V_0$] (0,3)
to[short, -*, i=$I_0$] (2,3)
to[R=$R_1$, i>_=$i_1$] (2,0) -- (0,0);
\draw (2,3) -- (4,3)
to[R=$R_2$, i>_=$i_2$]
(4,0) to[short, -*] (2,0);
\end{circuitikz}

\end{document}
```
````

<img width=375 align="right" src="./imgs/img3.png">

````latex
```tikz
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}

\begin{tikzpicture}
\begin{axis}[colormap/viridis]
\addplot3[
	surf,
	samples=18,
	domain=-3:3
]
{exp(-x^2-y^2)*x};
\end{axis}
\end{tikzpicture}

\end{document}
```
````

<img width=400 align="right" src="./imgs/img4.png">

````latex
```tikz
\usepackage{tikz-cd}

\begin{document}
\begin{tikzcd}

    T
    \arrow[drr, bend left, "x"]
    \arrow[ddr, bend right, "y"]
    \arrow[dr, dotted, "{(x,y)}" description] & & \\
    K & X \times_Z Y \arrow[r, "p"] \arrow[d, "q"]
    & X \arrow[d, "f"] \\
    & Y \arrow[r, "g"]
    & Z

\end{tikzcd}

\quad \quad

\begin{tikzcd}[row sep=2.5em]

A' \arrow[rr,"f'"] \arrow[dr,swap,"a"] \arrow[dd,swap,"g'"] &&
  B' \arrow[dd,swap,"h'" near start] \arrow[dr,"b"] \\
& A \arrow[rr,crossing over,"f" near start] &&
  B \arrow[dd,"h"] \\
C' \arrow[rr,"k'" near end] \arrow[dr,swap,"c"] && D' \arrow[dr,swap,"d"] \\
& C \arrow[rr,"k"] \arrow[uu,<-,crossing over,"g" near end]&& D

\end{tikzcd}

\end{document}
```
````

<img width=325 align="right" src="./imgs/img5.png">

````latex
```tikz
\usepackage{chemfig}
\begin{document}

\chemfig{[:-90]HN(-[::-45](-[::-45]R)=[::+45]O)>[::+45]*4(-(=O)-N*5(-(<:(=[::-60]O)-[::+60]OH)-(<[::+0])(<:[::-108])-S>)--)}

\end{document}
```
````

<img width=310 align="right" src="./imgs/img6.png">

````latex
```tikz
\usepackage{chemfig}
\begin{document}

\definesubmol\fragment1{

    (-[:#1,0.85,,,draw=none]
    -[::126]-[::-54](=_#(2pt,2pt)[::180])
    -[::-70](-[::-56.2,1.07]=^#(2pt,2pt)[::180,1.07])
    -[::110,0.6](-[::-148,0.60](=^[::180,0.35])-[::-18,1.1])
    -[::50,1.1](-[::18,0.60]=_[::180,0.35])
    -[::50,0.6]
    -[::110])
    }

\chemfig{
!\fragment{18}
!\fragment{90}
!\fragment{162}
!\fragment{234}
!\fragment{306}
}

\end{document}
```
````

## Function Plot Notes

When drawing mathematical functions in the current TikZJax environment, prefer conservative PGF math syntax. Some LaTeX/TikZ expressions that are valid in a full local LaTeX setup may fail to render here.

Practical notes from tested examples:

- Include `\begin{document}` and `\end{document}` inside every `tikz` block.
- For simple curves, use `\draw ... plot (\x,{...});`, matching the examples above.
- Prefer `\mathrm{...}` in labels instead of `\operatorname{...}` unless `\usepackage{amsmath}` is explicitly loaded.
- In PGF math expressions, write negative variables as `0-\x` when `exp(-\x)` fails.
- Avoid exponent syntax such as `(...)^2` if rendering fails; write multiplication explicitly, e.g. `(a)*(a)`.
- Start with no point markers. Commands such as `\filldraw`, `circle`, or open/closed point markers may fail depending on the renderer. Add them only after the base plot renders.
- If a function looks too flat, adjust `xscale` and `yscale` in `\begin{tikzpicture}` rather than changing the function.
- Add `samples=80` or `samples=100` for smooth curves.
- Avoid Chinese labels inside TikZ nodes in the current environment. They may cause rendering failure. Prefer English labels in the diagram, and explain Chinese terms in surrounding Markdown text if needed.

Stable sigmoid example:

````latex
```tikz
\begin{document}
\begin{tikzpicture}[domain=-6:6, xscale=0.75, yscale=3.0]
  \draw[very thin,color=gray!45] (-6.1,-0.1) grid (6.1,1.15);
  \draw[->] (-6.3,0) -- (6.4,0) node[right] {$x$};
  \draw[->] (0,-0.15) -- (0,1.25) node[above] {$f(x)$};

  \draw[dashed,color=gray] (-6,1) -- (6,1);
  \draw[dashed,color=gray] (-6,0.5) -- (6,0.5);
  \draw[dashed,color=gray] (-6,0.25) -- (6,0.25);

  \draw[color=red, thick, domain=-6:6, samples=100]
    plot (\x,{1/(1+exp(0-\x))});

  \draw[color=blue, thick, domain=-6:6, samples=100]
    plot (\x,{(1/(1+exp(0-\x)))*(1-(1/(1+exp(0-\x))))});

  \node[color=red,right] at (3.4,0.88) {$\mathrm{sigmoid}(x)$};
  \node[color=blue,right] at (1.0,0.25) {$\mathrm{sigmoid}'(x)$};

  \node[left] at (0,1) {$1$};
  \node[left] at (0,0.5) {$0.5$};
  \node[left] at (0,0.25) {$0.25$};
\end{tikzpicture}
\end{document}
```
````

## Contributing
Contributions are welcome! For information on building Tikzjax, have a look at the [contributing guide](https://github.com/artisticat1/obsidian-tikzjax/issues/68), courtesy of [@thecodechemist99](https://github.com/thecodechemist99).

## Acknowledgements
This plugin would not be possible without [TikZJax](https://github.com/kisonecat/tikzjax) by [@kisonecat](https://github.com/kisonecat)! In particular, it uses
[@drgrice1's fork](https://github.com/drgrice1/tikzjax/tree/ww-modifications) that adds some additional features.
