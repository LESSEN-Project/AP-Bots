from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        """
        Initialize the connection to the Neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """
        Close the connection when done.
        """
        self.driver.close()

    def add_or_update_user(self, user_id, properties):
        """
        Create or update a User node with the given properties.
        Timestamps (created_at and updated_at) are automatically set.
        Filters out None or empty values to avoid overwriting existing data with blanks.
        """
        # Filter out empty/None to preserve existing data
        filtered_props = {
            k: v for k, v in properties.items() 
            if v is not None and v != ""
        }

        with self.driver.session() as session:
            return session.execute_write(self._add_or_update_user_tx, user_id, filtered_props)

    @staticmethod
    def _add_or_update_user_tx(tx, user_id, properties):
        """
        MERGE on user_id, then set timestamps and update properties.
        """
        query = """
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET 
            u.created_at = datetime(), 
            u.updated_at = datetime()
        ON MATCH SET 
            u.updated_at = datetime()
        SET u += $properties
        RETURN u
        """
        result = tx.run(query, user_id=user_id, properties=properties)
        return result.single()[0]

    def update_user_profile_from_conversation(self, user_id, extracted_info):
        """
        Update the user's profile based on structured extraction.

        Example extracted_info structure:
        {
            "user": {
                "name": <string>,
                "surname": <string>,
                "age": <number or string>,
                "profession": <string>,
                "education": <string>,
                "ethnicity": <string>,
                "country_of_residence": <string>
            },
            "details": {
                "hobbies": [<string>, ...],
                "personality_traits": [<string>, ...]
            }
        }
        
        This method updates the User node and creates/updates:
          - Hobby nodes (relationship ENGAGES_IN)
          - PersonalityTrait nodes (relationship HAS_TRAIT)
        """
        print(extracted_info)
        # 1) Update User properties
        user_props = extracted_info.get("user", {})
        self.add_or_update_user(user_id, user_props)

        # 2) Process hobbies
        for hobby in extracted_info.get("details", {}).get("hobbies", []):
            hobby_id = hobby.lower().replace(" ", "_")
            self._create_or_update_hobby_and_relationship(user_id, hobby_id, hobby, "ENGAGES_IN")

        # 3) Process personality traits
        for trait in extracted_info.get("details", {}).get("personality_traits", []):
            trait_id = trait.lower().replace(" ", "_")
            self._create_or_update_personality_trait_and_relationship(user_id, trait_id, trait, "HAS_TRAIT")

        return extracted_info

    # -------------------- HOBBIES --------------------

    def _create_or_update_hobby_and_relationship(self, user_id, hobby_id, hobby_name, relation):
        """
        Create or update a Hobby node and link the User with ENGAGES_IN.
        """
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_hobby_tx, hobby_id, hobby_name)
            session.execute_write(
                self._add_relationship_tx,
                user_id, 
                relation, 
                "Hobby", 
                {"id": hobby_id, "name": hobby_name}, 
                {}
            )

    @staticmethod
    def _create_or_update_hobby_tx(tx, hobby_id, hobby_name):
        query = """
        MERGE (h:Hobby {id: $hobby_id})
        ON CREATE SET 
            h.created_at = datetime(), 
            h.updated_at = datetime()
        ON MATCH SET 
            h.updated_at = datetime()
        SET h.name = $hobby_name
        RETURN h
        """
        result = tx.run(query, hobby_id=hobby_id, hobby_name=hobby_name)
        return result.single()[0]

    # -------------------- PERSONALITY TRAITS --------------------

    def _create_or_update_personality_trait_and_relationship(self, user_id, trait_id, trait_name, relation):
        """
        Create or update a PersonalityTrait node and link the User with HAS_TRAIT.
        """
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_personality_trait_tx, trait_id, trait_name)
            session.execute_write(
                self._add_relationship_tx,
                user_id, 
                relation, 
                "PersonalityTrait", 
                {"id": trait_id, "name": trait_name}, 
                {}
            )

    @staticmethod
    def _create_or_update_personality_trait_tx(tx, trait_id, trait_name):
        query = """
        MERGE (pt:PersonalityTrait {id: $trait_id})
        ON CREATE SET 
            pt.created_at = datetime(), 
            pt.updated_at = datetime()
        ON MATCH SET 
            pt.updated_at = datetime()
        SET pt.name = $trait_name
        RETURN pt
        """
        result = tx.run(query, trait_id=trait_id, trait_name=trait_name)
        return result.single()[0]

    # -------------------- RELATIONSHIPS --------------------

    @staticmethod
    def _add_relationship_tx(tx, user_id, relation, target_label, target_properties, relation_props):
        """
        Create a relationship from the User node to a target node, with timestamps.
        """
        query = f"""
        MATCH (u:User {{user_id: $user_id}})
        MERGE (t:{target_label} {{id: $target_id}})
        ON CREATE SET t.created_at = datetime(), t.updated_at = datetime()
        ON MATCH SET t.updated_at = datetime()
        SET t += $target_properties
        MERGE (u)-[r:{relation}]->(t)
        ON CREATE SET r.created_at = datetime(), r.updated_at = datetime()
        ON MATCH SET r.updated_at = datetime()
        SET r += $relation_props
        RETURN u, r, t
        """
        params = {
            "user_id": user_id,
            "target_id": target_properties.get("id"),
            "target_properties": target_properties,
            "relation_props": relation_props
        }
        result = tx.run(query, **params)
        return result.single()

    # -------------------- QUERIES & DELETION --------------------

    def query_user_knowledge(self, user_id):
        """
        Retrieve all nodes and relationships connected to the given User.
        """
        with self.driver.session() as session:
            return session.execute_read(self._query_user_knowledge_tx, user_id)

    @staticmethod
    def _query_user_knowledge_tx(tx, user_id):
        query = """
        MATCH (u:User {user_id: $user_id})-[r]->(n)
        RETURN u, r, n
        """
        result = tx.run(query, user_id=user_id)
        return [record for record in result]

    def query_hobby_knowledge(self):
        """
        Retrieve all nodes and relationships connected to the given User.
        """
        with self.driver.session() as session:
            return session.execute_read(self._query_hobby_knowledge_tx)

    @staticmethod
    def _query_hobby_knowledge_tx(tx):
        query = """
        MATCH (h:Hobby)
        RETURN h
        """
        result = tx.run(query)
        return [record for record in result]

    def query_personality_knowledge(self):
        """
        Retrieve all nodes and relationships connected to the given User.
        """
        with self.driver.session() as session:
            return session.execute_read(self._query_personality_knowledge_tx)

    @staticmethod
    def _query_personality_knowledge_tx(tx):
        query = """
        MATCH (p:PersonalityTrait)
        RETURN p
        """
        result = tx.run(query)
        return [record for record in result]
        
    def delete_user(self, user_id):
        """
        Delete the User node (and all its relationships) from the graph.
        """
        with self.driver.session() as session:
            session.execute_write(self._delete_user_tx, user_id)

    @staticmethod
    def _delete_user_tx(tx, user_id):
        query = """
        MATCH (u:User {user_id: $user_id})
        DETACH DELETE u
        """
        tx.run(query, user_id=user_id)
