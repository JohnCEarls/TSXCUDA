# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.36
#
# Don't modify this file, modify the SWIG interface instead.

import _nvtsp
import new
new_instancemethod = new.instancemethod
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


def _swig_setattr_nondynamic_method(set):
    def set_attr(self,name,value):
        if (name == "thisown"): return self.this.own(value)
        if hasattr(self,name) or (name == "this"):
            set(self,name,value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr


class nvtsp_PySwigIterator(object):
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    def __init__(self, *args, **kwargs): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _nvtsp.delete_nvtsp_PySwigIterator
    __del__ = lambda self : None;
    def value(*args): return _nvtsp.nvtsp_PySwigIterator_value(*args)
    def incr(*args): return _nvtsp.nvtsp_PySwigIterator_incr(*args)
    def decr(*args): return _nvtsp.nvtsp_PySwigIterator_decr(*args)
    def distance(*args): return _nvtsp.nvtsp_PySwigIterator_distance(*args)
    def equal(*args): return _nvtsp.nvtsp_PySwigIterator_equal(*args)
    def copy(*args): return _nvtsp.nvtsp_PySwigIterator_copy(*args)
    def next(*args): return _nvtsp.nvtsp_PySwigIterator_next(*args)
    def previous(*args): return _nvtsp.nvtsp_PySwigIterator_previous(*args)
    def advance(*args): return _nvtsp.nvtsp_PySwigIterator_advance(*args)
    def __eq__(*args): return _nvtsp.nvtsp_PySwigIterator___eq__(*args)
    def __ne__(*args): return _nvtsp.nvtsp_PySwigIterator___ne__(*args)
    def __iadd__(*args): return _nvtsp.nvtsp_PySwigIterator___iadd__(*args)
    def __isub__(*args): return _nvtsp.nvtsp_PySwigIterator___isub__(*args)
    def __add__(*args): return _nvtsp.nvtsp_PySwigIterator___add__(*args)
    def __sub__(*args): return _nvtsp.nvtsp_PySwigIterator___sub__(*args)
    def __iter__(self): return self
nvtsp_PySwigIterator_swigregister = _nvtsp.nvtsp_PySwigIterator_swigregister
nvtsp_PySwigIterator_swigregister(nvtsp_PySwigIterator)

class IntVector(object):
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(*args): return _nvtsp.IntVector_iterator(*args)
    def __iter__(self): return self.iterator()
    def __nonzero__(*args): return _nvtsp.IntVector___nonzero__(*args)
    def __len__(*args): return _nvtsp.IntVector___len__(*args)
    def pop(*args): return _nvtsp.IntVector_pop(*args)
    def __getslice__(*args): return _nvtsp.IntVector___getslice__(*args)
    def __setslice__(*args): return _nvtsp.IntVector___setslice__(*args)
    def __delslice__(*args): return _nvtsp.IntVector___delslice__(*args)
    def __delitem__(*args): return _nvtsp.IntVector___delitem__(*args)
    def __getitem__(*args): return _nvtsp.IntVector___getitem__(*args)
    def __setitem__(*args): return _nvtsp.IntVector___setitem__(*args)
    def append(*args): return _nvtsp.IntVector_append(*args)
    def empty(*args): return _nvtsp.IntVector_empty(*args)
    def size(*args): return _nvtsp.IntVector_size(*args)
    def clear(*args): return _nvtsp.IntVector_clear(*args)
    def swap(*args): return _nvtsp.IntVector_swap(*args)
    def get_allocator(*args): return _nvtsp.IntVector_get_allocator(*args)
    def begin(*args): return _nvtsp.IntVector_begin(*args)
    def end(*args): return _nvtsp.IntVector_end(*args)
    def rbegin(*args): return _nvtsp.IntVector_rbegin(*args)
    def rend(*args): return _nvtsp.IntVector_rend(*args)
    def pop_back(*args): return _nvtsp.IntVector_pop_back(*args)
    def erase(*args): return _nvtsp.IntVector_erase(*args)
    def __init__(self, *args): 
        this = _nvtsp.new_IntVector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(*args): return _nvtsp.IntVector_push_back(*args)
    def front(*args): return _nvtsp.IntVector_front(*args)
    def back(*args): return _nvtsp.IntVector_back(*args)
    def assign(*args): return _nvtsp.IntVector_assign(*args)
    def resize(*args): return _nvtsp.IntVector_resize(*args)
    def insert(*args): return _nvtsp.IntVector_insert(*args)
    def reserve(*args): return _nvtsp.IntVector_reserve(*args)
    def capacity(*args): return _nvtsp.IntVector_capacity(*args)
    __swig_destroy__ = _nvtsp.delete_IntVector
    __del__ = lambda self : None;
IntVector_swigregister = _nvtsp.IntVector_swigregister
IntVector_swigregister(IntVector)

class DoubleVector(object):
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(*args): return _nvtsp.DoubleVector_iterator(*args)
    def __iter__(self): return self.iterator()
    def __nonzero__(*args): return _nvtsp.DoubleVector___nonzero__(*args)
    def __len__(*args): return _nvtsp.DoubleVector___len__(*args)
    def pop(*args): return _nvtsp.DoubleVector_pop(*args)
    def __getslice__(*args): return _nvtsp.DoubleVector___getslice__(*args)
    def __setslice__(*args): return _nvtsp.DoubleVector___setslice__(*args)
    def __delslice__(*args): return _nvtsp.DoubleVector___delslice__(*args)
    def __delitem__(*args): return _nvtsp.DoubleVector___delitem__(*args)
    def __getitem__(*args): return _nvtsp.DoubleVector___getitem__(*args)
    def __setitem__(*args): return _nvtsp.DoubleVector___setitem__(*args)
    def append(*args): return _nvtsp.DoubleVector_append(*args)
    def empty(*args): return _nvtsp.DoubleVector_empty(*args)
    def size(*args): return _nvtsp.DoubleVector_size(*args)
    def clear(*args): return _nvtsp.DoubleVector_clear(*args)
    def swap(*args): return _nvtsp.DoubleVector_swap(*args)
    def get_allocator(*args): return _nvtsp.DoubleVector_get_allocator(*args)
    def begin(*args): return _nvtsp.DoubleVector_begin(*args)
    def end(*args): return _nvtsp.DoubleVector_end(*args)
    def rbegin(*args): return _nvtsp.DoubleVector_rbegin(*args)
    def rend(*args): return _nvtsp.DoubleVector_rend(*args)
    def pop_back(*args): return _nvtsp.DoubleVector_pop_back(*args)
    def erase(*args): return _nvtsp.DoubleVector_erase(*args)
    def __init__(self, *args): 
        this = _nvtsp.new_DoubleVector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(*args): return _nvtsp.DoubleVector_push_back(*args)
    def front(*args): return _nvtsp.DoubleVector_front(*args)
    def back(*args): return _nvtsp.DoubleVector_back(*args)
    def assign(*args): return _nvtsp.DoubleVector_assign(*args)
    def resize(*args): return _nvtsp.DoubleVector_resize(*args)
    def insert(*args): return _nvtsp.DoubleVector_insert(*args)
    def reserve(*args): return _nvtsp.DoubleVector_reserve(*args)
    def capacity(*args): return _nvtsp.DoubleVector_capacity(*args)
    __swig_destroy__ = _nvtsp.delete_DoubleVector
    __del__ = lambda self : None;
DoubleVector_swigregister = _nvtsp.DoubleVector_swigregister
DoubleVector_swigregister(DoubleVector)

class BoolVector(object):
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(*args): return _nvtsp.BoolVector_iterator(*args)
    def __iter__(self): return self.iterator()
    def __nonzero__(*args): return _nvtsp.BoolVector___nonzero__(*args)
    def __len__(*args): return _nvtsp.BoolVector___len__(*args)
    def pop(*args): return _nvtsp.BoolVector_pop(*args)
    def __getslice__(*args): return _nvtsp.BoolVector___getslice__(*args)
    def __setslice__(*args): return _nvtsp.BoolVector___setslice__(*args)
    def __delslice__(*args): return _nvtsp.BoolVector___delslice__(*args)
    def __delitem__(*args): return _nvtsp.BoolVector___delitem__(*args)
    def __getitem__(*args): return _nvtsp.BoolVector___getitem__(*args)
    def __setitem__(*args): return _nvtsp.BoolVector___setitem__(*args)
    def append(*args): return _nvtsp.BoolVector_append(*args)
    def empty(*args): return _nvtsp.BoolVector_empty(*args)
    def size(*args): return _nvtsp.BoolVector_size(*args)
    def clear(*args): return _nvtsp.BoolVector_clear(*args)
    def swap(*args): return _nvtsp.BoolVector_swap(*args)
    def get_allocator(*args): return _nvtsp.BoolVector_get_allocator(*args)
    def begin(*args): return _nvtsp.BoolVector_begin(*args)
    def end(*args): return _nvtsp.BoolVector_end(*args)
    def rbegin(*args): return _nvtsp.BoolVector_rbegin(*args)
    def rend(*args): return _nvtsp.BoolVector_rend(*args)
    def pop_back(*args): return _nvtsp.BoolVector_pop_back(*args)
    def erase(*args): return _nvtsp.BoolVector_erase(*args)
    def __init__(self, *args): 
        this = _nvtsp.new_BoolVector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(*args): return _nvtsp.BoolVector_push_back(*args)
    def front(*args): return _nvtsp.BoolVector_front(*args)
    def back(*args): return _nvtsp.BoolVector_back(*args)
    def assign(*args): return _nvtsp.BoolVector_assign(*args)
    def resize(*args): return _nvtsp.BoolVector_resize(*args)
    def insert(*args): return _nvtsp.BoolVector_insert(*args)
    def reserve(*args): return _nvtsp.BoolVector_reserve(*args)
    def capacity(*args): return _nvtsp.BoolVector_capacity(*args)
    __swig_destroy__ = _nvtsp.delete_BoolVector
    __del__ = lambda self : None;
BoolVector_swigregister = _nvtsp.BoolVector_swigregister
BoolVector_swigregister(BoolVector)

runNVTSP = _nvtsp.runNVTSP

