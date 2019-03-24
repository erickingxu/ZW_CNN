#include "Layer.h"

namespace zwcnn {
	Layer::Layer()
	{
	}

	Layer::~Layer()
	{
	}

	int Layer::load_param(FILE* p)
	{
		return 0;
	}
	
	int Layer::load_param_bin(FILE* p) {
		return 0;
	}

	int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
	{
		return -1;
	}

#include "layer_declaration.h"

	static const layer_registry_entry layer_registry[] =
	{
#include "layer_registry.h"
	};
	static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);
	int layer_to_index(const char *type) {
		for (int i = 0; i<layer_registry_entry_count; i++)
		{
			if (strcmp(type, layer_registry[i].name) == 0)
			{
				return i;
			}
		}
		fprintf(stderr, "layer %s not exists\n", type);
	}

	Layer* create_layer(int index)
	{
		if (index < 0 || index >= layer_registry_entry_count)
		{
			fprintf(stderr, "layer index %d not exists\n", index);
			return 0;
		}

		layer_creator_func layer_creator = layer_registry[index].creator;
		if (!layer_creator)
		{
			fprintf(stderr, "layer index %d not enabled\n", index);
			return 0;
		}

		return layer_creator();
	}

}
