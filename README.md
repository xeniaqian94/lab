<!--ts-->



- [Installation](#installation)
  - [Using Yarn to install](#using-yarn-to-install)
- [Usage](#usage)
- [Dictionary](#dictionary)
- [Roadmap](#Roadmap)
- [Dev-Reference](#dev-reference)
- [Slate-TroubleShooting](#Slate-TroubleShooting)
  <!--te-->

# Installation

## Using Yarn to install

There are three steps:
1. `git clone [HTTP-project-url]`
2. Run `yarn`. It will install necessary packages
3. Run `yarn dev`. will compile the project and when it succeed, pop up the app


# Usage

On the canvas layout, the shortcut pattern is different for different platforms. 
- For Mac users, `fn+ctrl+[key]` is the shortcut pattern. For example, to connect a source node (claim/pdf, etc.) with one or some arbitrary target nodes, please select the source nodes, and `fn+ctrl+[right click]` on the target nodes. 
- For Windows users, `fn+ctrl+[key]` is the shortcut pattern. 

# Views

The right panel of the app has 3 views. Each view has different roles. 

- Canvas layout allows you to write and organize grounded claims
- Autograb list view shows a list of relevant sentences throughout the paper that fit your interest. The name "autograb" means these sentences are automatically grabbed by the text classification model. 
- DocEditor view is an wizard-of-oz interface to synthesis claims. As you writing, the interface will pops up a similar grounded claim that you have wrote before. 

For mac users, `ctrl+alt+[view_ID]` is the shortcut to switch between different views. 

# Dictionary

- **Auto-grab**: the feature that automatically grabs detailed information from the text content of the pdf.

# Build the application

To build the project into a standalone deskop application, run `yarn dist:mac`. 

By default, it will take 3-5 mins and once finished, will save a .zip file in the path `dist/compressor-0.0.1-mac.zip`. 


# Roadmap
Study 1: Do grounded claims improve sensemaking?
   - Subjects: ~40. undergrads? class credit? 
   - Task Constraints: under 1.5 hours. topics subjects can handle.
   - Task: Given our design and the default




-------
- ctrl-f
- transitions
   - text:    anything
   - table:   object attributes
   - network: relationships encoded with proximity, angle, containment, lines
   - map:     has named sequential axis


- clustering
   - initialize canvas coords with tsne

- chrome extension + firebase
   - select text, right click send to firebase list of {url, time, query}
   - firebase onlist add, knn search, list view 

- Fine-tuned, conditional semantic vectors for KNN search
   - Fine-tuned by machines
      - fastai + gorbid our pdfs?
   - conditional
      - paper/website title, authors, venue, section header
      - knowledge graph links

- document editor
   - insert one doc in another
   - transcullison?
   - paraphrasing something imported
   - create node from selected text
   - create nodes/links from outline heirarchy
   - insert viewbox into doc
   - change font size of any character
   - change font color
   - drag lines to create links between items in the editor

- queries -> search for anything, traverse the graph, large UI
   - query builder
      - a modal that take up entire window
      - autocomplete for search parameters
         - consider neo4j queries

- autocomplete ->  small UI
   - text matching algorithm
   - could be ghost text instead of popup 

- multi-instance customizable layouts
   - takes a query as input
   - canvas
   - list view
      - drag order
      - drag increase height
      - show links in margin?
      - add/remove/edit any node
      - drag links between
   - grid view
      - drag increase col/row size
      - show links in gutters?

- pdf viewer
   - manually fix autograb
   - good ole' fashion text highlighting
   - text highlighting from auto-grab
   - autocomplete for phrases in selected area
   - perf
      - zoom
      - prevent rerender
   
- graph viewer
   - select all left/right/up/down
   - move many with arrows
   - force directed
   - collision
   - alignment tools
   - html-to-image?
   - perf: react spring + transform + opacity

- Bug reports
   - text editor id can be undefined. maybe fixed?


# Slate-TroubleShooting
** Can't focus multiple instances of slate editors, e.g. in a list: **
Cause: conflicting keys used to find editor in the dom
Solution: editorUtils.tsx/initKeySafeSlate()

# Dev-Reference

# Sweet path from Matt's use 

This would be either field deployment inspiration or walk path for a demo video. 

process: random youtube rec -> watch 30 sec vid -> skim paper -> find interesting bit -> download into pdf dir -> open up electron -> grab bits to share -> screenshot -> paste into slack -> add context message in slack (edited)

# Design

- Predictive Interactions
  - easily dismissible, correctness, alternate possibilities, refinements
  - Read Eric Horvitz Principles of Mixed-Initiative User Interfaces

# CMD

ngrok http 3000 -host-header="localhost:3000"

# tours

https://reactour.js.org

# npm graphs

https://github.com/levelgraph/levelgraph 1200 # easy save to disk 3x slower than graphology for neighbors
https://github.com/graphology/graphology#readme 200 # 3x faster than levelgraph for neighbors
http://js.cytoscape.org/

# graph layouts

https://ialab.it.monash.edu/webcola/index.html
https://github.com/d3/d3-force

# electron perf notes

https://www.infoq.com/presentations/electron-pitfalls

# state management

https://github.com/CharlesStover/reactn
https://github.com/diegohaz/constate

# async validation

https://indicative.adonisjs.com/docs/syntax-guide

# hooks

https://github.com/neo/react-use-scroll-position
https://github.com/LeetCode-OpenSource/rxjs-hooks#apis

# node perf testing

https://github.com/bestiejs/platform.js#readme
https://benchmarkjs.com/

# async handling

https://github.com/staltz/callbag-basics # push and pull?
https://mostcore.readthedocs.io/en/latest/api.html # fastest
https://github.com/ReactiveX/rxjs

# formalism

level of detail/precision/specificity + attrs (esp. causality + utility)
broadcasting, static docs, cognitive tutors

# queries

what's the expected outcome of some scenario (what's y given x)
what scenario is optimal for creating some outcome (what predicts y)
what does the author mean by x (use expands to & same as )
how confident should I be in statement X

# fast static kdtree: not sorted by distance?

https://github.com/mourner/kdbush#readme

# DIY graph db

http://nodejsconfit.levelgraph.io/#17
