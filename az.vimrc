syntax on
set nocompatible              " be iMproved, required
filetype off                  " required
set backspace=2
set noswapfile
" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'Valloric/YouCompleteMe'
"Plugin 'prabirshrestha/async.vim'
"Plugin 'prabirshrestha/vim-lsp'
"Plugin 'prabirshrestha/asyncomplete.vim'
"Plugin 'prabirshrestha/asyncomplete-lsp.vim'
Plugin 'VundleVim/Vundle.vim'
Plugin 'hynek/vim-python-pep8-indent'
Plugin 'altercation/vim-colors-solarized'
Plugin 'tpope/vim-fugitive'
Plugin 'vim-syntastic/syntastic'
call vundle#end()            " required
filetype plugin indent on

" set statusline+=%#warningmsg#
" set statusline+=%{SyntasticStatuslineFlag()}
" set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0
let g:syntastic_python_flake8_post_args='--ignore=E501,E128,E225,F401'
let g:syntastic_quiet_messages = {'level': 'warnings'}
let g:ycm_autoclose_preview_window_after_completion=1
let g:ycm_autoclose_preview_window_after_completion=1

function! SwitchSourceHeader()
  update!
  if (expand ("%:e") == "cc")
    find %:t:r.h
  else
    find %:t:r.cc
  endif
endfunction
nmap ,s :call SwitchSourceHeader()<CR>

set laststatus=2
set tabstop=2
set shiftwidth=2
set smartindent
set expandtab
set clipboard=unnamed
set number
set wildmode=longest,list,full
set wildmenu

if has("autocmd")
  au BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$")
    \| exe "normal! g'\"" | endif
endif

nnoremap <silent> + :exe "vertical resize " . (winwidth(0) * 3/2)<CR>
nnoremap <silent> - :exe "vertical resize " . (winwidth(0) * 2/3)<CR>

" List characters
set list
set listchars=tab:>.,trail:.,extends:#,nbsp:.
" Colors
set t_Co=256
" solarized options
" let g:solarized_visibility = "high"
" let g:solarized_contrast = "high"
let g:solarized_termcolors = 256
set background=dark
colorscheme desert


" There's got to be a builtin for this!
function s:Min(num1, num2)
  if a:num1 < a:num2
    return a:num1
  endif
  return a:num2
endfunction

function s:Max(num1, num2)
  if a:num1 > a:num2
    return a:num1
  endif
  return a:num2
endfunction

fu! s:Sum(vals)
    let acc = 0
    for val in a:vals
        let acc += val
    endfor
    return acc
  endfu
fu! s:LogicalLineCounts()
    if &wrap
        let width = winwidth(0)
        let line_counts = map(range(1, line('$')), "foldclosed(v:val)==v:val?1:(virtcol([v:val, '$'])/width)+1")
    else
        let line_counts = [line('$')]
    endif
    return line_counts
endfu
fu! s:LinesHiddenByFoldsCount()
    let lines = range(1, line('$'))
    call filter(lines, "foldclosed(v:val) > 0 && foldclosed(v:val) != v:val")
    return len(lines)
endfu
fu! s:AutoResizeWindow(vert)
    if a:vert
        let longest = 100
        exec "set winwidth=" . longest
    else
        1
    endif
endfu
" Works when the sidebar is open
function! s:EnsureWidth()
  if winnr() == 1
    " close enough
    call s:AutoResizeWindow(1)
    wincmd =
  else
    call s:AutoResizeWindow(1)
    " Jump to window one, see if it's a NERDTree then resize it if so.
    wincmd t
    wincmd p  "then jump back to where we were
    wincmd =
  endif
endfunction

function! s:EnsureEqual()
  wincmd =
endfunction

:au BufEnter * call s:EnsureWidth()
:au WinEnter * call s:EnsureEqual()
:au TabEnter * call s:EnsureWidth()

" colorscheme desert
