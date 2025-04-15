# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:50:30 2021

@author: Quantum Engineer
"""

import numpy as np
import random
import string


def generate_wiki_table(n_rows, n_columns, content=None, 
                        row_headers=None, column_headers=None, 
                        plots=True, pixels=None):
    
    # prepare headers
    if row_headers:
        if not hasattr(row_headers, '__iter__'):
            row_headers = ['header' for _ in range(n_rows)]
    
    if column_headers:
        if not hasattr(column_headers, '__iter__'):
            column_headers = ['header' for _ in range(n_columns)]
            
        assert len(column_headers) == n_columns, f'If column headers are provided, the number of column headers ({len(column_headers)}) must match the number of columns ({n_columns})'
        
        if row_headers:
            assert len(row_headers) == n_rows, f'If row headers are provided, the number of row headers ({len(row_headers)}) must match the number of columns ({n_rows})'
            
            # add a dummy header for the header column
            column_headers.insert(0, '')
            
    # prepare content
    if content is not None:
        content_size = (len(content), len(content[0]))
        table_size = (n_rows, n_columns)
        
        assert content_size == table_size, f'Provided content is of size {content_size}, but the requested table is of size {table_size}'
        
    else:
        if plots:
            if pixels:
                assert type(pixels)==int, 'pixels kwarg must be an integer'
                content = f' |  {pixels}px]]'
            else:
                content = '  '
        else:
            content = 'content'
            
        content = [[content for _ in range(n_columns)] for _ in range(n_rows)]
    
    # initialize table
    table = '{| class="wikitable"\n'
    
    # add row of column headers
    if column_headers:
        table +=  '|-\n' + ''.join(f'! {header}\n' for header in column_headers)

    # fill table
    for row in range(n_rows):
        table += '|-\n'
        
        if row_headers:
            table += f'! {row_headers[row]}\n'
        
        for column in range(n_columns):
            table += f'| {content[row][column]}\n'
            
    table += '|}'
    
    return table

class TableElement():
    def __init__(self, text, row=None, column=None, row_label=None, column_label=None):
        self.text = text
        self.row = row
        self.column = column
        self.row_label = row_label
        self.column_label = column_label
        self.formatted_text = None
        
        
    def format_text(self, plot=False, pixels=None):
        if not plot:
            self.formatted_text = self.text
        
        else:
            if pixels is not None:
                self.formatted_text = f'[[File: {self.text} | {pixels}px]]'
            else:
                self.formatted_text = f'[[File: {self.text}]]'
                
        return self.formatted_text

class WikiTable():
    def __init__(self, n_rows=None, n_columns=None):
        self.n_rows = n_rows
        self.n_columns = n_columns        
        self.elements = []
        self.table = None
    
    def add_element(self, content, row=None, column=None, row_label=None, column_label=None):
        if row is None:
            row = len(self.elements)
            
        if column is None:
            column = 0
            
        self.elements.append(TableElement(content, row, column, row_label, column_label))
        
    def remove_element(self, idx):
        self.elements.pop(idx)
        
    def generate(self, pixels=None, auto_stack='rows'):
        
        assert auto_stack in [None, 'rows', 'columns'], 'Keyword argument "auto_stacking" must be either "columns" or "rows" or None'
        self.auto_stack = auto_stack
        
        # assign indices
        if self.n_rows and self.n_columns:
            n_rows, n_columns = self.n_rows, self.n_columns
            for e in self.elements:
                assert (e.row is not None) and (e.column is not None), 'If n_rows and n_columns are provided, all elements must be assigned a row and column'
            
        elif self.auto_stack == 'rows':
            n_rows, n_columns = len(self.elements), 1
            for idx, e in enumerate(self.elements):
                e.row = idx
                e.column = 0
            
        elif self.auto_stack =='columns':
            n_rows, n_columns = 1, len(self.elements)
            for idx, e in enumerate(self.elements):
                e.row = 0
                e.column = idx
        
        # populate class table
        content = [[None for _ in range(n_columns)] for _ in range(n_rows)]
        for e in self.elements:
            content[e.row][e.column] = e
                
        # populate text table
        text_content = [[e.format_text(plot=e.text.endswith('.png'), pixels=pixels) for e in row] for row in content]
        
        # pull headers
        row_headers = [content[r][0].row_label for r in range(n_rows)]
        column_headers = [content[0][c].column_label for c in range(n_columns)]
        
        if all(header is None for header in row_headers):
            row_headers = None
        if all(header is None for header in column_headers):
            column_headers = None
        
        self.table = generate_wiki_table(n_rows, n_columns, text_content, row_headers, column_headers)
        
        return self.table
    
    def print(self):
        assert self.table is not None, 'We have not generated the table yet. Run the generate method first.'
        print(self.table)
    
if __name__ == '__main__':
    
    n_rows = 3
    n_columns = 4
    
    # user-defined table
    # content = np.random.rand(n_rows, n_columns) 
    # content = None
    # column_headers = None# [f'column {idx}' for idx in range(n_columns)]
    # row_headers = None#[f'row {idx}' for idx in range(n_rows)]
    
    # plots = True
    # pixels = 500
    
    # table = generate_wiki_table(n_rows, n_columns, content, row_headers, column_headers, plots, pixels)
    # print(table)

    # auto-generated table
    table = WikiTable()
    
    for r in range(n_rows):
        for c in range(n_columns):
            filename = ''.join(random.sample(string.ascii_lowercase, 6)) + '.png'
            row_header = str(r)
            column_header = str(c)
            table.add_element(filename)
            
    table.generate(pixels=500, auto_stack='rows')
    table.print()