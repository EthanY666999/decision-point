"""Catch deferred entrypoint NameErrors without loading GPU dependencies."""
import builtins
import symtable
import unittest
from pathlib import Path

class DriverScope(unittest.TestCase):
    def test_function_globals_resolve(self):
        path=Path(__file__).resolve().parents[1]/'train_v4.py'
        table=symtable.symtable(path.read_text(),str(path),'exec')
        defined={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        defined.update(dir(builtins));defined.update({'__file__','__name__'})
        def check(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global():
                    self.assertIn(symbol.get_name(),defined,(scope.get_name(),symbol.get_name()))
            for child in scope.get_children():check(child)
        for scope in table.get_children():check(scope)

if __name__=='__main__':unittest.main()
