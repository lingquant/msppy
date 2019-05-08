#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import gurobipy
def _copy_uncertainty_rhs(value, target, attribute):
    """Copy rhs uncertainty (attribute, value) to target"""
    result = {}
    for constr_tuple, value in value.items():
        if type(constr_tuple) == tuple:
            s = [
                target._model.getConstrByName(x.constrName)
                for x in constr_tuple
            ]
            result[tuple(s)] = value
        else:
            s = target._model.getConstrByName(constr_tuple.constrName)
            result[s] = value
    setattr(target, attribute, result)

def _copy_uncertainty_coef(value, target, attribute):
    """Copy coef uncertainty (attribute, value) to target"""
    result = {}
    for key, value in value.items():
        constr = target._model.getConstrByName(key[0].constrName)
        var = target._model.getVarByName(key[1].varName)
        result[(constr, var)] = value
    setattr(target, attribute, result)

def _copy_uncertainty_obj(value, target, attribute):
    """Copy obj uncertainty (attribute, value) to target"""
    result = {}
    for var_tuple, value in value.items():
        if type(var_tuple) == tuple:
            s = [target._model.getVarByName(x.varName) for x in var_tuple]
            result[tuple(s)] = value
        else:
            s = target._model.getVarByName(var_tuple.varName)
            result[s] = value
    setattr(target, attribute, result)

def _copy_uncertainty_mix(value, target, attribute):
    """Copy mixed uncertainty (attribute, value) to target"""
    result = {}
    for keys, dist in value.items():
        s = []
        for key in keys:
            if type(key) == gurobipy.Var:
                s.append(target._model.getVarByName(key.varName))
            elif type(key) == gurobipy.Constr:
                s.append(target._model.getConstrByName(key.constrName))
            else:
                constr = target._model.getConstrByName(key[0].constrName)
                var = target._model.getVarByName(key[1].varName)
                s.append((constr, var))
        result[tuple(s)] = dist
    setattr(target, attribute, result)

def _copy_vars(value, target, attribute):
    """Copy vars (attribute, value) to target"""
    if type(value) == list:
        result = [target._model.getVarByName(x.varName) for x in value]
    else:
        result = (
            target._model.getVarByName(value.varName)
            if value is not None
            else None
        )
    setattr(target, attribute, result)

def _copy_constrs(value, target, attribute):
    """Copy constrs (attribute, value) to target"""
    if type(value) == list:
        result = [
            target._model.getConstrByName(x.constrName) for x in value
        ]
    else:
        result = target._model.getConstrByName(value.constrName)
    setattr(target, attribute, result)
