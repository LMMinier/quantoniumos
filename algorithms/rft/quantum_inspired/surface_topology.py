#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Surface topology utilities.

Provides triangulated surface generators and invariant computation
(Euler characteristic, orientability, genus/crosscap number).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class SurfaceTopology:
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    orientable: bool
    genus: Optional[int]
    crosscap_number: Optional[int]


@dataclass(frozen=True)
class TriangulatedSurface:
    vertex_count: int
    faces: List[Tuple[int, int, int]]


def _build_edge_map(faces: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]]:
    edge_map: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append((face_index, (u, v)))
    return edge_map


def _is_orientable(faces: List[Tuple[int, int, int]]) -> bool:
    edge_map = _build_edge_map(faces)
    adjacency: Dict[int, List[Tuple[int, int]]] = {}

    for entries in edge_map.values():
        if len(entries) != 2:
            continue
        (face_a, dir_a), (face_b, dir_b) = entries
        same_direction = dir_a == dir_b
        relation = -1 if same_direction else 1
        adjacency.setdefault(face_a, []).append((face_b, relation))
        adjacency.setdefault(face_b, []).append((face_a, relation))

    orientation: Dict[int, int] = {}
    for face_index in range(len(faces)):
        if face_index in orientation:
            continue
        orientation[face_index] = 1
        stack = [face_index]
        while stack:
            current = stack.pop()
            for neighbor, relation in adjacency.get(current, []):
                expected = orientation[current] * relation
                if neighbor not in orientation:
                    orientation[neighbor] = expected
                    stack.append(neighbor)
                elif orientation[neighbor] != expected:
                    return False
    return True


def compute_surface_topology(surface: TriangulatedSurface) -> SurfaceTopology:
    edge_map = _build_edge_map(surface.faces)
    vertex_count = surface.vertex_count
    edge_count = len(edge_map)
    face_count = len(surface.faces)
    euler_characteristic = vertex_count - edge_count + face_count
    orientable = _is_orientable(surface.faces)

    genus: Optional[int] = None
    crosscap_number: Optional[int] = None
    if orientable:
        g = (2 - euler_characteristic) / 2
        if abs(g - round(g)) < 1e-9 and g >= 0:
            genus = int(round(g))
    else:
        k = 2 - euler_characteristic
        if abs(k - round(k)) < 1e-9 and k >= 1:
            crosscap_number = int(round(k))

    return SurfaceTopology(
        vertex_count=vertex_count,
        edge_count=edge_count,
        face_count=face_count,
        euler_characteristic=euler_characteristic,
        orientable=orientable,
        genus=genus,
        crosscap_number=crosscap_number,
    )


def triangulate_torus(nu: int, nv: int) -> TriangulatedSurface:
    if nu < 3 or nv < 3:
        raise ValueError("Torus triangulation requires nu,nv >= 3")

    vertex_ids: Dict[Tuple[int, int], int] = {}

    def vid(i: int, j: int) -> int:
        key = (i % nu, j % nv)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertex_ids)
        return vertex_ids[key]

    faces: List[Tuple[int, int, int]] = []
    for i in range(nu):
        for j in range(nv):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i, j + 1)
            d = vid(i + 1, j + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))

    return TriangulatedSurface(vertex_count=len(vertex_ids), faces=faces)


def _klein_key(i: int, j: int, nu: int, nv: int) -> Tuple[int, int]:
    if j == nv:
        i = (nu - i) % nu
        j = 0
    i = i % nu
    j = j % nv
    return i, j


def triangulate_klein_bottle(nu: int, nv: int) -> TriangulatedSurface:
    if nu < 3 or nv < 3:
        raise ValueError("Klein bottle triangulation requires nu,nv >= 3")

    vertex_ids: Dict[Tuple[int, int], int] = {}

    def vid(i: int, j: int) -> int:
        key = _klein_key(i, j, nu, nv)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertex_ids)
        return vertex_ids[key]

    faces: List[Tuple[int, int, int]] = []
    for i in range(nu):
        for j in range(nv):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i, j + 1)
            d = vid(i + 1, j + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))

    return TriangulatedSurface(vertex_count=len(vertex_ids), faces=faces)
