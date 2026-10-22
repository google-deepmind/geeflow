# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module deprecated. Please use geeflow.data_sources instead."""

from geeflow.data_sources import boundaries
from geeflow.data_sources import climate
from geeflow.data_sources import custom
from geeflow.data_sources import demographics
from geeflow.data_sources import ee_base
from geeflow.data_sources import elevation
from geeflow.data_sources import land_cover
from geeflow.data_sources import lidar
from geeflow.data_sources import mapbiomas
from geeflow.data_sources import modis
from geeflow.data_sources import optical
from geeflow.data_sources import radar

EeData = ee_base.EeData
EeDataFC = ee_base.EeDataFC

# custom.py
CustomFC = custom.CustomFC
CustomImage = custom.CustomImage
CustomIC = custom.CustomIC

# elevation.py
NasaDem = elevation.NasaDem
FABDEM = elevation.FABDEM
CopDem = elevation.CopDem

# radar.py
Sentinel1 = radar.Sentinel1
Alos = radar.Alos

# optical.py
Sentinel2 = optical.Sentinel2
HLS = optical.HLS
LandsatComposite = optical.LandsatComposite
Landsat7 = optical.Landsat7
Landsat8 = optical.Landsat8
Landsat9 = optical.Landsat9
Landsat5 = optical.Landsat5
NAIP = optical.NAIP
Nicfi = optical.Nicfi
LandsatTimelapse = optical.LandsatTimelapse

# modis.py
ModisTerraVeg = modis.ModisTerraVeg
ModisSurfRefl = modis.ModisSurfRefl
ModisGPP = modis.ModisGPP
ModisET = modis.ModisET
ModisBurn = modis.ModisBurn
ModisFire = modis.ModisFire
FIRMS = modis.FIRMS
ModisLaiFpar = modis.ModisLaiFpar

# land_cover.py
LandCover = land_cover.LandCover
WSF2015 = land_cover.WSF2015
TreeCoverLossDueToFire = land_cover.TreeCoverLossDueToFire
Hansen = land_cover.Hansen
RaddAlert = land_cover.RaddAlert
Primary = land_cover.Primary
NLCD = land_cover.NLCD
DynamicWorld = land_cover.DynamicWorld
WorldCover = land_cover.WorldCover
CCDC = land_cover.CCDC

# lidar.py
GediRasterCanopyHeight = lidar.GediRasterCanopyHeight
GediRasterCanopyStructure = lidar.GediRasterCanopyStructure
GediRasterAGB = lidar.GediRasterAGB

# demographics.py
FPP = demographics.FPP
TPP = demographics.TPP
CIESIN = demographics.CIESIN
WorldPop = demographics.WorldPop
GHSPop = demographics.GHSPop

# boundaries.py
Ecoregions = boundaries.Ecoregions
Countries = boundaries.Countries
RPGFrance2023 = boundaries.RPGFrance2023

# climate.py
Era5 = climate.Era5

# mapbiomas.py
MapBiomasLandsat = mapbiomas.MapBiomasLandsat
MapBiomasAtmosphere = mapbiomas.MapBiomasAtmosphere
MapBiomasSoil = mapbiomas.MapBiomasSoil
MapBiomasSoilCarbon = mapbiomas.MapBiomasSoilCarbon
MapBiomasFire = mapbiomas.MapBiomasFire
MapBiomasLulcBrazil = mapbiomas.MapBiomasLulcBrazil
