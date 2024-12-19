resource "azurerm_resource_group" "adl_rg" {
  name     = "RG-NeilSinclair"
  location = var.location
  tags     = var.tags
}